(function () {
  // ============================================================
  // IIFE (Immediately Invoked Function Expression)
  // ============================================================
  // - Hàm tự chạy ngay khi file JS load.
  // - Mục tiêu:
  //   + Tạo scope riêng tránh đụng biến global
  //   + Code các pipeline crawl khác nhau (HTML/API/SELENIUM) không bị trùng biến/hàm
  // - Những gì gắn vào window.* (vd: window.clearLog) mới lộ ra global.
  //
  // Lưu ý: Vì chạy ngay, script thường đặt cuối body hoặc đảm bảo DOM đã có sẵn.
  // (Bạn yêu cầu không đổi logic nên chỉ chú thích)
  const cfg = window.__VRAIN_SELENIUM__ || {};
  // cfg: object config backend inject vào window để JS biết endpoint nào cần gọi.
  // Thường cfg có:
  // - cfg.startUrl: endpoint POST để start selenium job
  // - cfg.logsUrl: endpoint GET để poll logs theo job_id + offset
  // Ví dụ backend render:
  // window.__VRAIN_SELENIUM__ = { startUrl: "/crawl-vrain-selenium/start/", logsUrl: "/crawl-vrain-selenium/logs/" }

  // ============================================================
  // LẤY ELEMENTS UI
  // ============================================================
  const logBox = document.getElementById("log-container"); // vùng hiển thị log realtime
  const btn = document.getElementById("btnStartCrawl");    // nút Start crawl
  const spinner = document.getElementById("spinner");      // spinner loading (hiển thị khi đang chạy)
  const statusValue = document.getElementById("statusValue"); // text trạng thái (đang chạy / sẵn sàng)
  const lastCrawlTime = document.getElementById("lastCrawlTime"); // thời gian crawl gần nhất
  const lastFileSize = document.getElementById("lastFileSize");   // size file output gần nhất

  // ============================================================
  // BIẾN TRẠNG THÁI CHO CƠ CHẾ JOB + POLLING LOGS
  // ============================================================
  let jobId = null;
  // jobId:
  // - Selenium crawl thường chạy "job" ở backend (có thể tốn thời gian)
  // - Backend trả về job_id để frontend poll logs đúng job đó
  // - Nếu không có jobId => pollLogs không chạy (tránh poll nhầm)

  let offset = 0;
  // offset:
  // - dùng để lấy log incremental (tương tự cursor)
  // - Backend trả offset mới sau mỗi lần poll
  // - Lần sau chỉ lấy phần log mới từ offset đó -> tiết kiệm tải

  let timer = null;
  // timer:
  // - lưu id của setInterval
  // - để dừng polling khi job done hoặc khi restart

  // ============================================================
  // setRunningUI(isRunning): UPDATE UI THEO TRẠNG THÁI JOB
  // ============================================================
  // - isRunning=true:
  //   + show spinner
  //   + status: "Đang chạy..."
  //   + disable nút start
  // - isRunning=false:
  //   + hide spinner
  //   + status: "Sẵn sàng"
  //   + enable nút start
  function setRunningUI(isRunning) {
    if (spinner) spinner.style.display = isRunning ? "inline-block" : "none";
    if (statusValue) statusValue.textContent = isRunning ? "🔄 Đang chạy..." : "✅ Sẵn sàng";
    if (btn) btn.disabled = isRunning;
  }

  // ============================================================
  // appendLines(lines): THÊM DÒNG LOG MỚI VÀO UI
  // ============================================================
  // - Append (thêm) để giữ lịch sử log
  // - Xoá placeholder muted khi bắt đầu có log thật
  // - Auto scroll xuống cuối
  function appendLines(lines) {
    if (!logBox || !lines || lines.length === 0) return;

    // Dòng muted là dòng placeholder dạng: "Đang chạy… log sẽ cập nhật realtime."
    const muted = logBox.querySelector(".log__line--muted");
    if (muted) muted.remove();

    // Append từng line thành div log__line
    for (const line of lines) {
      const div = document.createElement("div");
      div.className = "log__line";
      div.textContent = line; // dùng textContent để tránh inject HTML
      logBox.appendChild(div);
    }

    // scroll xuống cuối để theo dõi realtime
    logBox.scrollTop = logBox.scrollHeight;
  }

  // ============================================================
  // getCookie(name): LẤY COOKIE THEO TÊN (DÙNG CHO CSRF)
  // ============================================================
  // - Django thường lưu CSRF token trong cookie "csrftoken"
  // - POST request cần gửi header "X-CSRFToken" để pass CSRF middleware
  // - Cách làm:
  //   + prefix "; " để dễ split
  //   + split theo `; name=`
  //   + nếu có đúng 2 phần => lấy phần sau và cắt đến dấu ";"
  function getCookie(name) {
    const v = `; ${document.cookie}`;
    const parts = v.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(";").shift();
    return "";
  }

  // ============================================================
  // startJob(): START SELENIUM JOB Ở BACKEND
  // ============================================================
  // Luồng:
  // 1) Check cfg.startUrl
  // 2) setRunningUI(true)
  // 3) POST startUrl kèm CSRF
  // 4) Backend trả JSON:
  //    { ok: true, job_id: "...", ... }
  // 5) Reset offset=0, set jobId
  // 6) Reset logBox + bật polling logs
  async function startJob() {
    // Nếu thiếu startUrl => không thể start
    if (!cfg.startUrl) return alert("Thiếu startUrl.");

    // UI chuyển sang trạng thái đang chạy
    setRunningUI(true);

    try {
      // POST start job
      const res = await fetch(cfg.startUrl, {
        method: "POST",
        headers: {
          // Gửi CSRF token trong header để Django cho phép POST
          "X-CSRFToken": getCookie("csrftoken")
        }
      });

      // Cố parse JSON; nếu parse fail thì fallback {}
      // (tránh crash khi backend trả HTML/empty)
      const data = await res.json().catch(() => ({}));

      // Nếu HTTP fail hoặc backend trả ok=false => throw lỗi
      // - data.message ưu tiên hiển thị message backend
      // - fallback "HTTP <status>"
      if (!res.ok || !data.ok) throw new Error(data.message || ("HTTP " + res.status));

      // Lưu job_id để pollLogs biết đang poll job nào
      jobId = data.job_id;

      // Reset offset để lấy log từ đầu (hoặc theo logic backend)
      offset = 0;

      // Reset log box hiển thị placeholder muted
      if (logBox) {
        logBox.innerHTML = '<div class="log__line log__line--muted">Đang chạy… log sẽ cập nhật realtime.</div>';
      }

      // Nếu đang poll từ lần chạy trước -> clear interval để tránh poll chồng
      if (timer) clearInterval(timer);

      // Poll logs mỗi 900ms
      timer = setInterval(pollLogs, 900);

      // Poll ngay lập tức để có log sớm
      await pollLogs();
    } catch (e) {
      // Nếu start job lỗi:
      // - UI về trạng thái sẵn sàng
      // - alert lỗi cho user/dev
      setRunningUI(false);
      alert("Start crawl lỗi: " + (e?.message || e));
    }
  }

  // ============================================================
  // pollLogs(): POLL LOGS THEO job_id + offset
  // ============================================================
  // Luồng:
  // 1) Check cfg.logsUrl và jobId (phải có jobId mới poll được)
  // 2) Tạo URL với query:
  //    - job_id=<jobId>
  //    - offset=<offset>
  // 3) GET logsUrl
  // 4) Backend trả JSON:
  //    {
  //      ok: true,
  //      lines: [...],
  //      offset: <new_offset>,
  //      done: true/false,
  //      last_crawl_time: "...",
  //      last_size_mb: ...
  //    }
  // 5) Append lines, cập nhật offset, cập nhật UI, dừng polling nếu done
  async function pollLogs() {
    // Nếu thiếu logsUrl hoặc chưa có jobId => không poll
    if (!cfg.logsUrl || !jobId) return;

    try {
      // Tạo URL object để dễ set searchParams
      const url = new URL(cfg.logsUrl, window.location.origin);

      // Gắn job_id để backend biết cần lấy log của job nào
      url.searchParams.set("job_id", jobId);

      // Gắn offset để backend trả log incremental từ offset hiện tại
      url.searchParams.set("offset", String(offset));

      // GET logs
      const res = await fetch(url.toString(), { method: "GET" });
      if (!res.ok) return; // nếu HTTP fail -> im lặng return

      // Parse JSON, fallback {}
      const data = await res.json().catch(() => ({}));
      if (!data.ok) return; // backend báo ok=false -> bỏ qua

      // Append log mới vào UI
      appendLines(data.lines || []);

      // Update offset mới do backend trả về (nếu có)
      offset = data.offset ?? offset;

      // Update UI: last crawl time
      if (data.last_crawl_time && lastCrawlTime) lastCrawlTime.textContent = data.last_crawl_time;

      // Update UI: last file size
      if (typeof data.last_size_mb !== "undefined" && lastFileSize) {
        lastFileSize.textContent = data.last_size_mb ? `${data.last_size_mb} MB` : "–";
      }

      // setRunningUI dựa vào data.done:
      // - done=false => vẫn đang chạy => isRunning=true
      // - done=true  => job xong => isRunning=false
      setRunningUI(!data.done);

      // Nếu job done -> dừng polling
      if (data.done && timer) {
        clearInterval(timer);
        timer = null;
      }
    } catch (e) {
      // Catch trống:
      // - Nuốt lỗi để tránh spam alert khi polling fail tạm thời
      // - Trong dev có thể console.error(e) nếu muốn debug
      // (Bạn yêu cầu không đổi logic nên giữ nguyên)
    }
  }

  // ============================================================
  // window.clearLog: HÀM XOÁ LOG BOX (GLOBAL)
  // ============================================================
  // - Gắn vào window để gọi từ HTML (onclick="clearLog()")
  // - Reset log box về trạng thái muted "Log đã được xoá."
  window.clearLog = function () {
    if (!logBox) return;
    logBox.innerHTML = '<div class="log__line log__line--muted">Log đã được xoá.</div>';
  };

  // ============================================================
  // GẮN EVENT CLICK NÚT START
  // ============================================================
  // - Khi bấm nút start => gọi startJob()
  if (btn) btn.addEventListener("click", startJob);
})(); // Kết thúc IIFE
