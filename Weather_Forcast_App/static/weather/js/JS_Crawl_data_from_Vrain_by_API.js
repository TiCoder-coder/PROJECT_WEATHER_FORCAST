(function () {
  // ============================================================
  // IIFE (Immediately Invoked Function Expression)
  // ============================================================
  // - Đây là 1 hàm tự chạy ngay khi file JS được load.
  // - Mục tiêu:
  //   + Tạo "scope" riêng, tránh làm bẩn biến global (window)
  //   + Các biến/hàm nội bộ không bị trùng tên với file JS khác
  // - Chỉ những gì bạn gắn vào window.* thì mới dùng được ở bên ngoài.
  //
  // Lưu ý: Vì code chạy ngay, nó giả định:
  // - Script được đặt sau HTML (cuối body) hoặc DOM đã sẵn sàng.
  // (Bạn yêu cầu không đổi logic nên chỉ ghi chú)
  const cfg = window.__VRAIN_API__ || {};
  // cfg là object config mà backend inject vào window để JS dùng:
  // Ví dụ backend render:
  // window.__VRAIN_API__ = {
  //   startUrl: "/crawl-vrain-api/start/",
  //   tailUrl: "/crawl-vrain-api/logs/",
  //   csrfToken: "...."  (tuỳ)
  // }
  //
  // Nếu backend chưa inject thì cfg = {} để tránh crash.

  // ============================================================
  // LẤY CÁC ELEMENT TRÊN UI
  // ============================================================
  const logBox = document.getElementById("log-container"); // khung hiển thị log realtime
  const btn = document.getElementById("btnStartCrawl");    // nút "Start crawl"
  const spinner = document.getElementById("spinner");      // spinner loading
  const statusValue = document.getElementById("statusValue"); // text trạng thái
  const lastCrawlTime = document.getElementById("lastCrawlTime"); // thời gian crawl gần nhất
  const lastFileSize = document.getElementById("lastFileSize");   // size file output gần nhất

  // ============================================================
  // BIẾN TRẠNG THÁI PHỤC VỤ POLLING
  // ============================================================
  let since = 0;
  // since: mốc "offset/cursor" để tail log incremental
  // - Backend sẽ trả log mới kể từ since và trả next_since
  // - Giúp tiết kiệm băng thông: không tải toàn bộ log mỗi lần

  let timer = null;
  // timer: lưu id của setInterval để stop polling khi job kết thúc hoặc restart job

  // ============================================================
  // getCookie(name): LẤY COOKIE THEO TÊN (DÙNG CHO CSRF)
  // ============================================================
  // - Một số trường hợp cfg.csrfToken không có (backend không inject),
  //   ta fallback lấy "csrftoken" từ cookie của Django.
  // - document.cookie dạng: "a=1; csrftoken=xyz; b=2"
  // - Ta split("; ") thành từng cặp "key=value"
  // - decodeURIComponent để xử lý ký tự encode trong cookie value
  function getCookie(name) {
    const parts = document.cookie ? document.cookie.split("; ") : [];
    for (const part of parts) {
      const [k, v] = part.split("=");
      if (k === name) return decodeURIComponent(v || "");
    }
    return "";
  }

  // ============================================================
  // setRunningUI(isRunning): CẬP NHẬT UI THEO TRẠNG THÁI JOB
  // ============================================================
  // isRunning = true:
  // - show spinner
  // - status: "Đang chạy..."
  // - disable nút start để tránh click nhiều lần
  //
  // isRunning = false:
  // - hide spinner
  // - status: "Sẵn sàng"
  // - enable nút start để user có thể chạy lại
  function setRunningUI(isRunning) {
    if (spinner) spinner.style.display = isRunning ? "inline-block" : "none";
    if (statusValue) statusValue.textContent = isRunning ? "🔄 Đang chạy..." : "✅ Sẵn sàng";
    if (btn) btn.disabled = isRunning;
  }

  // ============================================================
  // appendLines(lines): THÊM DÒNG LOG MỚI VÀO LOG BOX
  // ============================================================
  // - Append (thêm) chứ không replace để user xem được lịch sử log
  // - Nếu logBox có dòng placeholder muted thì xoá đi khi có log thật
  // - Auto scroll xuống cuối để user luôn thấy log mới nhất
  function appendLines(lines) {
    if (!logBox || !lines || lines.length === 0) return;

    // Xoá placeholder "muted" nếu tồn tại (vd: "Đang chạy…")
    const muted = logBox.querySelector(".log__line--muted");
    if (muted) muted.remove();

    // Append từng dòng log thành 1 div
    for (const line of lines) {
      const div = document.createElement("div");
      div.className = "log__line";
      div.textContent = line; // an toàn (không render HTML)
      logBox.appendChild(div);
    }

    // Scroll xuống cuối để theo dõi realtime
    logBox.scrollTop = logBox.scrollHeight;
  }

  // ============================================================
  // startJob(): BẮT ĐẦU JOB CRAWL (POST) + BẬT POLLING LOGS
  // ============================================================
  // Luồng:
  // 1) Check cfg.startUrl tồn tại
  // 2) setRunningUI(true)
  // 3) POST startUrl (kèm CSRF)
  // 4) Reset since = 0
  // 5) Reset logBox hiển thị placeholder
  // 6) Start setInterval pollLogs mỗi 900ms và poll ngay lập tức
  async function startJob() {
    // Nếu thiếu startUrl -> không thể start job
    if (!cfg.startUrl) {
      alert("Thiếu startUrl.");
      return;
    }

    // Update UI sang trạng thái đang chạy
    setRunningUI(true);

    try {
      // Lấy CSRF token:
      // - Ưu tiên cfg.csrfToken (backend inject)
      // - Nếu không có -> lấy từ cookie "csrftoken" (chuẩn Django)
      const csrf = cfg.csrfToken || getCookie("csrftoken");

      // Gửi POST request start job
      const res = await fetch(cfg.startUrl, {
        method: "POST",

        // credentials: "same-origin"
        // - gửi cookie session
        // - cần nếu start endpoint yêu cầu login/session auth
        credentials: "same-origin",

        headers: {
          // CSRF header để Django cho phép POST
          "X-CSRFToken": csrf,

          // Body JSON nên set content-type json
          "Content-Type": "application/json",
        },

        // Body rỗng: backend chỉ cần biết "start"
        body: JSON.stringify({}),
      });

      // Nếu HTTP fail (4xx/5xx):
      // - đọc response text để có thông báo lỗi chi tiết
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || ("HTTP " + res.status));
      }

      // Start thành công -> reset cursor logs
      since = 0;

      // Clear log box và show placeholder muted
      if (logBox) {
        logBox.innerHTML =
          '<div class="log__line log__line--muted">Đang chạy… log sẽ cập nhật realtime.</div>';
      }

      // Nếu đang có interval cũ -> clear để tránh poll chồng
      if (timer) clearInterval(timer);

      // Bắt đầu poll logs mỗi 900ms (~0.9s)
      timer = setInterval(pollLogs, 900);

      // Poll ngay lập tức để user thấy log nhanh hơn
      await pollLogs();
    } catch (e) {
      // Nếu start lỗi:
      // - UI quay về trạng thái sẵn sàng
      // - alert lỗi cho user/dev
      setRunningUI(false);
      alert("Start crawl lỗi: " + (e?.message || e));
    }
  }

  // ============================================================
  // pollLogs(): LẤY LOG MỚI TỪ BACKEND (GET TAIL)
  // ============================================================
  // Luồng:
  // 1) Check cfg.tailUrl
  // 2) Tạo URL có query param since=<cursor>
  // 3) GET tailUrl
  // 4) Parse JSON -> data
  // 5) Nếu data.ok:
  //    + appendLines(data.lines)
  //    + since = data.next_since
  //    + cập nhật last_crawl_time, last_size_mb
  //    + setRunningUI(data.is_running)
  //    + nếu job kết thúc -> clearInterval(timer)
  async function pollLogs() {
    // Nếu không có tailUrl -> không thể poll
    if (!cfg.tailUrl) return;

    try {
      // Tạo URL object để set query params dễ dàng
      const url = new URL(cfg.tailUrl, window.location.origin);

      // since là cursor offset để backend trả log incremental
      url.searchParams.set("since", String(since));

      // GET logs
      const res = await fetch(url.toString(), {
        method: "GET",
        credentials: "same-origin", // gửi cookie/session
      });
      if (!res.ok) return; // nếu lỗi HTTP -> im lặng return

      // Parse JSON
      const data = await res.json();

      // Convention backend:
      // - data.ok: true/false
      // - nếu ok=false thì bỏ qua (không update UI)
      if (!data.ok) return;

      // Append log mới
      appendLines(data.lines || []);

      // Cập nhật cursor cho lần poll tiếp theo
      since = data.next_since ?? since;

      // Update UI info "last crawl time"
      if (data.last_crawl_time && lastCrawlTime) lastCrawlTime.textContent = data.last_crawl_time;

      // Update UI "last file size"
      if (typeof data.last_size_mb !== "undefined" && lastFileSize) {
        // Nếu có size -> show "<mb> MB"
        // Nếu không có/0/null -> show "–"
        lastFileSize.textContent = data.last_size_mb ? `${data.last_size_mb} MB` : "–";
      }

      // Update UI running theo backend trả về
      setRunningUI(!!data.is_running);

      // Nếu job đã kết thúc -> dừng polling
      if (!data.is_running && timer) {
        clearInterval(timer);
        timer = null;
      }
    } catch (e) {
      // Catch trống:
      // - Bạn chọn "nuốt" lỗi để tránh spam UI/alert khi polling fail tạm thời
      // - Trong dev có thể console.error(e) để debug
      // (Bạn yêu cầu không đổi logic nên giữ nguyên)
    }
  }

  // ============================================================
  // window.clearLog: XOÁ LOG TRÊN UI (HÀM GLOBAL)
  // ============================================================
  // - Gắn vào window để có thể gọi từ HTML (onclick="clearLog()")
  // - Reset logBox và show placeholder "Log đã được xoá."
  window.clearLog = function () {
    if (!logBox) return;
    logBox.innerHTML = '<div class="log__line log__line--muted">Log đã được xoá.</div>';
  };

  // ============================================================
  // GẮN EVENT CLICK CHO NÚT START
  // ============================================================
  // - Khi click nút start -> gọi startJob()
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
})(); // Kết thúc IIFE, chạy ngay khi file JS load
 