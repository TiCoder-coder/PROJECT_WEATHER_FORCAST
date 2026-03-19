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

  function setQueuedUI(queuePosition) {
    if (spinner) spinner.style.display = "inline-block";
    if (statusValue) statusValue.textContent = `🕐 Đang đợi trong hàng — vị trí #${queuePosition}`;
    if (btn) btn.disabled = true;
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

      // Update offset mới (hỗ trợ cả offset (compat) lẫn next_since (mới))
      offset = data.offset ?? data.next_since ?? offset;

      if (data.last_crawl_time && lastCrawlTime) lastCrawlTime.textContent = data.last_crawl_time;
      if (typeof data.last_size_mb !== "undefined" && lastFileSize) {
        lastFileSize.textContent = data.last_size_mb ? `${data.last_size_mb} MB` : "–";
      }

      // Nếu job đang xếp hàng -> show vị trí và tiếp tục poll
      if (data.is_queued) {
        setQueuedUI(data.queue_position || 0);
        return;
      }

      // setRunningUI dựa vào is_running (có compat done từ backend cũ)
      setRunningUI(!!(data.is_running || (data.done === false)));

      // Dừng polling khi job không còn chạy và không còn xếp hàng
      const finished = !data.is_running && !data.is_queued;
      if (finished && timer) {
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

  // ============================================================
  // AUTO-CRAWL: TỰ ĐỘNG LẶP LẠI THU THẬP THEO CHU KỲ
  // ============================================================
  const autoCrawlPanel = document.getElementById("autoCrawlPanel");
  const btnAutoStart = document.getElementById("btnAutoStart");
  const btnAutoStop = document.getElementById("btnAutoStop");
  const autoCrawlInterval = document.getElementById("autoCrawlInterval");
  const autoCrawlUnit = document.getElementById("autoCrawlUnit");
  const autoCrawlStatus = document.getElementById("autoCrawlStatus");
  const autoCrawlStatusText = document.getElementById("autoCrawlStatusText");
  const autoCrawlProgress = document.getElementById("autoCrawlProgress");
  const autoCrawlProgressFill = document.getElementById("autoCrawlProgressFill");
  const autoCrawlCountdown = document.getElementById("autoCrawlCountdown");
  const autoCrawlRound = document.getElementById("autoCrawlRound");

  let autoActive = false;
  let autoRound = 0;
  let autoCountdownTimer = null;
  let autoWaitTimer = null;

  function getIntervalMs() {
    const val = Math.max(1, parseInt(autoCrawlInterval.value, 10) || 5);
    const unit = autoCrawlUnit.value;
    return unit === "hours" ? val * 3600000 : val * 60000;
  }

  function formatCountdown(ms) {
    const totalSec = Math.max(0, Math.ceil(ms / 1000));
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return m > 0 ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
  }

  function setAutoUI(active) {
    if (btnAutoStart) btnAutoStart.disabled = active;
    if (btnAutoStop) btnAutoStop.disabled = !active;
    if (autoCrawlInterval) autoCrawlInterval.disabled = active;
    if (autoCrawlUnit) autoCrawlUnit.disabled = active;
    if (autoCrawlPanel) {
      autoCrawlPanel.classList.toggle("auto-crawl--active", active);
    }
    if (autoCrawlStatus) {
      autoCrawlStatus.className = active
        ? "auto-crawl__status auto-crawl__status--active"
        : "auto-crawl__status";
    }
  }

  function startAutoCountdown(callback) {
    const totalMs = getIntervalMs();
    let remaining = totalMs;
    const started = Date.now();

    if (autoCrawlProgress) autoCrawlProgress.style.display = "flex";
    if (autoCrawlProgressFill) autoCrawlProgressFill.style.width = "0%";

    autoCountdownTimer = setInterval(function () {
      remaining = totalMs - (Date.now() - started);
      if (remaining <= 0) {
        clearInterval(autoCountdownTimer);
        autoCountdownTimer = null;
        if (autoCrawlProgressFill) autoCrawlProgressFill.style.width = "100%";
        if (autoCrawlCountdown) autoCrawlCountdown.textContent = "0s";
        callback();
        return;
      }
      const pct = ((totalMs - remaining) / totalMs) * 100;
      if (autoCrawlProgressFill) autoCrawlProgressFill.style.width = pct.toFixed(1) + "%";
      if (autoCrawlCountdown) autoCrawlCountdown.textContent = formatCountdown(remaining);
    }, 1000);

    if (autoCrawlCountdown) autoCrawlCountdown.textContent = formatCountdown(totalMs);
  }

  function waitForJobDone(callback) {
    autoWaitTimer = setInterval(function () {
      if (!btn || !btn.disabled) {
        clearInterval(autoWaitTimer);
        autoWaitTimer = null;
        callback();
      }
    }, 500);
  }

  function runAutoRound() {
    if (!autoActive) return;
    autoRound++;
    if (autoCrawlRound) autoCrawlRound.textContent = "(#" + autoRound + ")";
    if (autoCrawlStatusText) autoCrawlStatusText.textContent = "🔄 Đang thu thập...";
    if (autoCrawlProgress) autoCrawlProgress.style.display = "none";

    startJob();

    waitForJobDone(function () {
      if (!autoActive) return;
      if (autoCrawlStatusText) {
        autoCrawlStatusText.textContent = "⏳ Chờ lần tiếp theo...";
      }
      startAutoCountdown(runAutoRound);
    });
  }

  function startAuto() {
    autoActive = true;
    autoRound = 0;
    setAutoUI(true);
    runAutoRound();
  }

  function stopAuto() {
    autoActive = false;
    if (autoCountdownTimer) { clearInterval(autoCountdownTimer); autoCountdownTimer = null; }
    if (autoWaitTimer) { clearInterval(autoWaitTimer); autoWaitTimer = null; }
    setAutoUI(false);
    if (autoCrawlStatusText) autoCrawlStatusText.textContent = "⏸ Đã dừng tự động";
    if (autoCrawlProgress) autoCrawlProgress.style.display = "none";
    if (autoCrawlProgressFill) autoCrawlProgressFill.style.width = "0%";
    if (autoCrawlRound) autoCrawlRound.textContent = "";
  }

  if (btnAutoStart) btnAutoStart.addEventListener("click", startAuto);
  if (btnAutoStop) btnAutoStop.addEventListener("click", stopAuto);
})(); // Kết thúc IIFE
