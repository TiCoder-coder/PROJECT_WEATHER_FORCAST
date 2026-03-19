console.log("✅ JS_Crawl_data_by_API.js loaded"); // Log để xác nhận file JS này đã được load (debug nhanh trên Console)

document.addEventListener("DOMContentLoaded", () => {
  // ============================================================
  // KHỐI 1: LẤY CÁC ELEMENT TRÊN DOM
  // ============================================================
  // form: form chính có class "form" (nơi user bấm submit để start crawl)
  // logContainer: khu vực hiển thị logs (div#log-container)
  // startBtn: nút submit cụ thể có name="action" value="start"
  //          (giúp ta disable/enable và đổi text khi job đang chạy)
  const form = document.querySelector("form.form");
  const logContainer = document.getElementById("log-container");
  const startBtn = document.querySelector('button[type="submit"][name="action"][value="start"]');

  // ============================================================
  // KHỐI 2: KHAI BÁO ENDPOINTS (ROUTES BACKEND)
  // ============================================================
  // START_URL:
  // - Endpoint backend để "start job crawl"
  // - Thường Django view nhận POST + action=start rồi chạy pipeline
  const START_URL = "/crawl-api-weather/";

  // LOGS_URL:
  // - Endpoint backend trả logs dạng JSON cho frontend poll liên tục
  // - Dùng để cập nhật log real-time (giả lập real-time bằng polling)
  const LOGS_URL = "/crawl-api-weather/logs/";

  // pollTimer:
  // - Biến lưu id của setInterval
  // - Dùng để dừng polling khi job kết thúc hoặc khi start mới
  let pollTimer = null;

  let currentJobId = null;
  // currentJobId: lưu job_id trả về từ backend sau khi start job

  let since = 0;
  // since: mốc cursor để tail log incremental (chỉ lấy log mới)

  // ============================================================
  // KHỐI 3: HÀM HIỂN THỊ LOGS LÊN UI (INCREMENTAL)
  // ============================================================
  // appendLines(lines):
  // - Thêm dòng log MỚI vào cuối logContainer (không xóa log cũ)
  // - Auto scroll xuống cuối để user luôn thấy log mới nhất
  function appendLines(lines) {
    if (!logContainer || !lines || lines.length === 0) return;

    // Xoá placeholder "muted" nếu tồn tại
    const muted = logContainer.querySelector(".log__line--muted");
    if (muted) muted.remove();

    // Append từng dòng log thành 1 div
    for (const line of lines) {
      const div = document.createElement("div");
      div.className = "log__line";
      div.textContent = line;
      logContainer.appendChild(div);
    }

    // Auto scroll đến cuối để luôn thấy log mới
    logContainer.scrollTop = logContainer.scrollHeight;
  }

  // ============================================================
  // KHỐI 4: HÀM FETCH LOGS TỪ BACKEND
  // ============================================================
  // fetchLogs():
  // - Gửi GET đến LOGS_URL
  // - Yêu cầu backend trả JSON (application/json)
  // - Nếu backend trả HTML (ví dụ bị redirect, lỗi 500, trả template)
  //   => throw Error để setLog hiển thị lỗi rõ ràng
  async function fetchLogs() {
    const url = new URL(LOGS_URL, window.location.origin);
    url.searchParams.set("since", String(since));
    if (currentJobId) url.searchParams.set("job_id", currentJobId);

    const res = await fetch(url.toString(), {
      // Header này giúp backend biết đây là request AJAX
      // (Django hay check X-Requested-With để trả JSON thay vì render template)
      headers: { "X-Requested-With": "XMLHttpRequest" },

      // cache: "no-store" để tránh trình duyệt cache logs cũ
      cache: "no-store",

      // credentials: "same-origin"
      // - gửi cookie session (nếu dùng login/session auth)
      // - cần thiết nếu endpoint logs yêu cầu đăng nhập
      credentials: "same-origin",
    });

    // Kiểm tra content-type:
    // - nếu không phải JSON => lấy text để debug (cắt 120 ký tự đầu)
    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      const text = await res.text();
      throw new Error("Logs endpoint không trả JSON. Response: " + text.slice(0, 120));
    }

    // Parse JSON và return object
    return await res.json();
  }

  // ============================================================
  // KHỐI 5: HÀM START JOB (GỬI POST)
  // ============================================================
  // startJob(formData):
  // - Gửi POST lên START_URL kèm FormData
  // - Mong backend trả JSON dạng:
  //   { ok: true, ... } hoặc { ok: false, error: "..." }
  // - Nếu trả không phải JSON => throw lỗi để debug rõ
  async function startJob(formData) {
    const res = await fetch(START_URL, {
      method: "POST",
      body: formData, // FormData chứa input của form + action=start
      headers: { "X-Requested-With": "XMLHttpRequest" }, // báo backend đây là AJAX
      credentials: "same-origin", // gửi cookie/session
    });

    // Validate content-type trả về
    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      const text = await res.text();
      throw new Error("Start endpoint không trả JSON. Response: " + text.slice(0, 120));
    }

    // Parse JSON
    const data = await res.json();

    // Nếu backend báo ok=false => throw để vào catch
    if (!data.ok) throw new Error(data.error || "start failed");

    // Trả data thành công cho caller
    return data;
  }

  // Nếu không có form => script không chạy (tránh lỗi trang khác không có form)
  if (!form) return;

  // ============================================================
  // KHỐI 6: XỬ LÝ SUBMIT FORM -> START + POLL LOGS
  // ============================================================
  form.addEventListener("submit", async (e) => {
    // Chặn submit truyền thống (reload trang)
    // Vì ta muốn submit bằng AJAX để giữ UI + cập nhật log real-time
    e.preventDefault();

    // Lấy dữ liệu từ form UI
    const formData = new FormData(form);

    // Ép action=start (để backend biết đây là lệnh start job)
    // - có thể backend dùng request.POST["action"] để phân nhánh
    formData.set("action", "start");

    // Disable nút start để tránh user bấm nhiều lần tạo nhiều job cùng lúc
    if (startBtn) {
      startBtn.disabled = true;
      startBtn.textContent = "⏳ Đang chạy... vui lòng chờ"; // feedback UX
    }

    // Nếu đang có polling từ lần chạy trước => clear để tránh setInterval chồng nhau
    if (pollTimer) clearInterval(pollTimer);

    try {
      // Lưu job_id để truyền vào fetchLogs
      const startData = await startJob(formData);
      currentJobId = startData.job_id || null;
      since = 0;

      // Clear log box và show placeholder
      if (logContainer) {
        logContainer.innerHTML =
          '<div class="log__line log__line--muted">Đang chạy… log sẽ cập nhật realtime.</div>';
      }

      // ============================================================
      // 2) FETCH LOGS LẦN ĐẦU ĐỂ HIỂN THỊ NGAY
      // ============================================================
      const first = await fetchLogs();

      // (Cố gắng cập nhật size file CSV mới nhất lên UI)
      const sizeEl = document.getElementById("lastFileSize");
      if (sizeEl && first.csv_size_mb != null) sizeEl.textContent = `${first.csv_size_mb} MB`;

      // Render logs lần đầu (incremental)
      appendLines(first.lines || []);
      since = first.next_since ?? since;

      // ============================================================
      // 3) BẮT ĐẦU POLLING LOGS MỖI 1 GIÂY
      // ============================================================
      // Mục tiêu:
      // - liên tục gọi fetchLogs()
      // - cập nhật UI logs
      // - khi backend báo is_running=false thì stop polling + enable nút start lại
      pollTimer = setInterval(async () => {
        try {
          const d = await fetchLogs();

          // Nếu job đang xếp hàng -> show vị trí và tiếp tục poll
          if (d.is_queued) {
            const pos = d.queue_position || 0;
            const statusEl = document.getElementById("statusValue");
            if (statusEl) statusEl.textContent = `🕐 Đang đợi trong hàng — vị trí #${pos}`;
            return;
          }

          appendLines(d.lines || []);
          since = d.next_since ?? since;

          if (!d.is_running && !d.is_queued) {
            // dừng polling
            clearInterval(pollTimer);
            pollTimer = null;

            // enable lại nút start + đổi text về ban đầu
            if (startBtn) {
              startBtn.disabled = false;
              startBtn.textContent = "🚀 Bắt đầu crawl ngay";
            }
          }
        } catch (err) {
          // Nếu fetchLogs lỗi (mất mạng, backend lỗi, trả html...)
          // -> hiển thị 1 dòng lỗi lên log UI
          appendLines([`[ERROR] ${err.message}`]);
        }
      }, 1000); // 1000ms = 1 giây
    } catch (err) {
      // Nếu startJob lỗi (hoặc bước trước đó lỗi):
      // - hiển thị lỗi lên UI log
      appendLines([`[ERROR] ${err.message}`]);

      // - enable lại nút start để user thử lại
      if (startBtn) {
        startBtn.disabled = false;
        startBtn.textContent = "🚀 Bắt đầu crawl ngay";
      }
    }
  });

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
      if (!startBtn || !startBtn.disabled) {
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

    // Build FormData from the form and trigger submit programmatically
    const formData = new FormData(form);
    formData.set("action", "start");

    if (startBtn) {
      startBtn.disabled = true;
      startBtn.textContent = "⏳ Đang chạy... vui lòng chờ";
    }
    if (pollTimer) clearInterval(pollTimer);
    since = 0;

    // Clear log box cho round mới
    if (logContainer) {
      logContainer.innerHTML =
        '<div class="log__line log__line--muted">Đang chạy… log sẽ cập nhật realtime.</div>';
    }

    startJob(formData).then(function (startData) {
      currentJobId = startData.job_id || null;
      return fetchLogs();
    }).then(function (first) {
      const sizeEl = document.getElementById("lastFileSize");
      if (sizeEl && first.csv_size_mb != null) sizeEl.textContent = first.csv_size_mb + " MB";
      appendLines(first.lines || []);
      since = first.next_since ?? since;

      pollTimer = setInterval(async function () {
        try {
          const d = await fetchLogs();

          if (d.is_queued) {
            const pos = d.queue_position || 0;
            const statusEl = document.getElementById("statusValue");
            if (statusEl) statusEl.textContent = `🕐 Đang đợi trong hàng — vị trí #${pos}`;
            return;
          }

          appendLines(d.lines || []);
          since = d.next_since ?? since;
          if (!d.is_running && !d.is_queued) {
            clearInterval(pollTimer);
            pollTimer = null;
            if (startBtn) {
              startBtn.disabled = false;
              startBtn.textContent = "🚀 Bắt đầu crawl ngay";
            }
          }
        } catch (err) {
          appendLines(["[ERROR] " + err.message]);
        }
      }, 1000);
    }).catch(function (err) {
      appendLines(["[ERROR] " + err.message]);
      if (startBtn) {
        startBtn.disabled = false;
        startBtn.textContent = "🚀 Bắt đầu crawl ngay";
      }
    });

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
});
