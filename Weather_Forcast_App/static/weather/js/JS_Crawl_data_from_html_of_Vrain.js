// ============================================================
// getCookie(name): HÀM LẤY GIÁ TRỊ COOKIE THEO TÊN
// ============================================================
// Mục đích:
// - Trên các app dùng Django/CSRF, đôi khi bạn cần lấy csrftoken từ cookie
// - Hàm này dùng regex match trong document.cookie để lấy value
//
// Cách hoạt động:
// - document.cookie là chuỗi dạng: "a=1; csrftoken=xyz; b=2"
// - regex sẽ tìm name=... và lấy nhóm giá trị
function getCookie(name) {
  // match: tìm chuỗi dạng: (^|;) <spaces> name <spaces>=<spaces> (value không chứa ;)
  const v = document.cookie.match("(^|;)\\s*" + name + "\\s*=\\s*([^;]+)");
  // Nếu match được -> v.pop() trả group cuối cùng (chính là value)
  // Nếu không -> trả chuỗi rỗng
  return v ? v.pop() : "";
}

(function () {
  // ============================================================
  // IIFE (Immediately Invoked Function Expression)
  // ============================================================
  // - Đây là "hàm tự chạy" ngay khi file JS load
  // - Mục tiêu: tạo scope riêng tránh pollute global namespace
  // - Các biến/hàm bên trong không bị lộ ra global (trừ những gì bạn gắn vào window)
  //
  // Lưu ý: đoạn code này KHÔNG chờ DOMContentLoaded,
  // nên nó giả định script được đặt ở cuối body hoặc DOM đã sẵn sàng.
  // (Bạn yêu cầu không đổi logic nên chỉ ghi chú)
  const cfg = window.__VRAIN_HTML__ || {};
  // cfg thường được backend inject vào HTML để truyền config động:
  // - cfg.startUrl: endpoint start crawl
  // - cfg.tailUrl: endpoint lấy logs (tail)
  // - cfg.csrfToken: token CSRF (nếu backend inject)
  //
  // Ví dụ backend render:
  // <script>window.__VRAIN_HTML__ = { startUrl: "...", tailUrl: "...", csrfToken: "..." }</script>

  // ============================================================
  // LẤY ELEMENTS UI TRÊN TRANG
  // ============================================================
  const logBox = document.getElementById("log-container"); // nơi hiển thị log (dạng list các dòng)
  const btn = document.getElementById("btnStartCrawl");    // nút bấm "Start crawl"
  const spinner = document.getElementById("spinner");      // icon/spinner báo đang chạy
  const statusValue = document.getElementById("statusValue"); // text trạng thái (đang chạy / sẵn sàng)
  const lastCrawlTime = document.getElementById("lastCrawlTime"); // text thời gian crawl gần nhất
  const lastFileSize = document.getElementById("lastFileSize");   // text size file output gần nhất

  // ============================================================
  // BIẾN TRẠNG THÁI CHO CƠ CHẾ POLLING LOGS
  // ============================================================
  let since = 0;
  // since:
  // - dùng để "tail" log incremental
  // - backend sẽ trả:
  //   + lines: các dòng log mới kể từ since
  //   + next_since: mốc mới để lần sau fetch tiếp
  //
  // Tương tự kiểu offset / cursor để không phải tải toàn bộ log mỗi lần.

  let timer = null;
  // timer:
  // - lưu id của setInterval để stop polling khi job kết thúc hoặc start lại

  // ============================================================
  // setRunningUI(isRunning): CẬP NHẬT UI THEO TRẠNG THÁI JOB
  // ============================================================
  // - isRunning = true:
  //   + show spinner
  //   + status "Đang chạy..."
  //   + disable nút start để tránh click nhiều lần
  // - isRunning = false:
  //   + hide spinner
  //   + status "Sẵn sàng"
  //   + enable nút start
  function setRunningUI(isRunning) {
    if (spinner) spinner.style.display = isRunning ? "inline-block" : "none";
    if (statusValue) statusValue.textContent = isRunning ? "🔄 Đang chạy..." : "✅ Sẵn sàng";
    if (btn) btn.disabled = isRunning;
  }

  // ============================================================
  // appendLines(lines): THÊM CÁC DÒNG LOG MỚI VÀO LOG BOX
  // ============================================================
  // Mục tiêu:
  // - Append (thêm) log mới thay vì replace toàn bộ log
  // - Giữ log cũ để user xem lại
  // - Auto scroll xuống cuối
  function appendLines(lines) {
    if (!logBox) return;                 // nếu không có log box -> bỏ qua
    if (!lines || lines.length === 0) return; // nếu không có dòng mới -> không làm gì

    // Nếu đang có dòng "muted" (placeholder kiểu: "Đang chạy…")
    // thì xoá đi khi bắt đầu có log thật
    const muted = logBox.querySelector(".log__line--muted");
    if (muted) muted.remove();

    // Append từng line thành một <div class="log__line">
    for (const line of lines) {
      const div = document.createElement("div");
      div.className = "log__line";
      div.textContent = line; // dùng textContent tránh injection HTML/XSS
      logBox.appendChild(div);
    }

    // Auto scroll xuống cuối để luôn thấy log mới nhất
    logBox.scrollTop = logBox.scrollHeight;
  }

  // ============================================================
  // startJob(): GỬI REQUEST START CRAWL (POST) -> BẮT ĐẦU POLLING LOG
  // ============================================================
  // Luồng:
  // 1) Check cfg.startUrl
  // 2) Set UI running
  // 3) POST startUrl (JSON body)
  // 4) Reset since = 0
  // 5) Reset log box hiển thị placeholder
  // 6) Start polling (setInterval) + pollLogs() ngay lập tức
  async function startJob() {
    // Nếu backend chưa inject startUrl thì báo lỗi
    if (!cfg.startUrl) {
      alert("Thiếu startUrl.");
      return;
    }

    // Set UI sang trạng thái đang chạy
    setRunningUI(true);

    try {
      // Gửi POST đến startUrl để backend khởi chạy pipeline crawl
      const res = await fetch(cfg.startUrl, {
        method: "POST",
        headers: {
          // CSRF:
          // - Django yêu cầu CSRF token với POST (nếu dùng CSRF protection)
          // - cfg.csrfToken được backend inject (hoặc có thể lấy từ cookie qua getCookie)
          // - Ở đây code dùng cfg.csrfToken, nếu thiếu sẽ gửi chuỗi rỗng
          "X-CSRFToken": cfg.csrfToken || "",
          // Content-Type JSON vì body là JSON.stringify
          "Content-Type": "application/json",
        },
        // body rỗng {}: chỉ báo "start" và backend tự xử lý
        body: JSON.stringify({}),
      });

      // Nếu HTTP không OK (4xx/5xx) -> đọc text để show lỗi rõ
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || ("HTTP " + res.status));
      }

      // Reset since về 0 để lần poll đầu lấy log từ đầu (hoặc theo logic backend)
      since = 0;

      // Clear log box và show placeholder muted
      if (logBox) {
        logBox.innerHTML = '<div class="log__line log__line--muted">Đang chạy… log sẽ cập nhật realtime.</div>';
      }

      // Nếu đang có interval cũ -> clear để tránh polling chồng
      if (timer) clearInterval(timer);

      // Bắt đầu polling logs mỗi 900ms (~0.9 giây)
      timer = setInterval(pollLogs, 900);

      // Poll ngay lập tức để có log sớm (không phải chờ 900ms)
      await pollLogs();
    } catch (e) {
      // Nếu start lỗi:
      // - Set UI về không chạy
      // - Alert lỗi cho user/dev
      setRunningUI(false);
      alert("Start crawl lỗi: " + (e?.message || e));
    }
  }

  // ============================================================
  // pollLogs(): LẤY LOGS MỚI TỪ BACKEND (TAIL)
  // ============================================================
  // Luồng:
  // 1) Check cfg.tailUrl
  // 2) Tạo URL có query param since=<offset>
  // 3) GET endpoint
  // 4) Parse JSON
  // 5) Append lines mới + cập nhật since = next_since
  // 6) Update UI: last_crawl_time, last_size_mb, is_running
  // 7) Nếu job kết thúc -> clearInterval(timer)
  async function pollLogs() {
    // Nếu không có tailUrl thì không thể poll logs
    if (!cfg.tailUrl) return;

    try {
      // Tạo URL object để dễ set searchParams
      const url = new URL(cfg.tailUrl, window.location.origin);

      // since=<mốc offset>: backend dùng để trả log mới kể từ mốc này
      url.searchParams.set("since", String(since));

      // Fetch logs (GET)
      const res = await fetch(url.toString(), { method: "GET" });
      if (!res.ok) return; // nếu lỗi HTTP thì im lặng return (không phá UI)

      // Parse JSON từ backend
      const data = await res.json();

      // Backend convention:
      // - data.ok = true/false
      // - nếu ok=false -> return im lặng
      if (!data.ok) return;

      // Append các dòng log mới vào UI
      appendLines(data.lines || []);

      // Cập nhật since theo next_since backend trả về
      // - dùng ?? để nếu next_since null/undefined thì giữ nguyên since cũ
      since = data.next_since ?? since;

      // Cập nhật thời gian crawl gần nhất nếu backend có trả
      if (data.last_crawl_time && lastCrawlTime) lastCrawlTime.textContent = data.last_crawl_time;

      // Cập nhật size file (MB) nếu backend có trả last_size_mb
      // - typeof !== "undefined" để phân biệt trường hợp backend không gửi field này
      if (typeof data.last_size_mb !== "undefined" && lastFileSize) {
        // Nếu last_size_mb truthy -> hiển thị "<mb> MB"
        // Nếu falsy (0/null/""...) -> hiển thị "–"
        lastFileSize.textContent = data.last_size_mb ? `${data.last_size_mb} MB` : "–";
      }

      // Update UI running theo data.is_running từ backend
      // - !! ép về boolean
      setRunningUI(!!data.is_running);

      // Nếu job đã kết thúc:
      // - stop polling để tiết kiệm request
      if (!data.is_running && timer) {
        clearInterval(timer);
        timer = null;
      }
    } catch (e) {
      // Catch trống:
      // - Bạn đang chọn "nuốt" lỗi polling để tránh spam alert/log
      // - Trong dev, bạn có thể console.error(e) để debug
      // (Bạn yêu cầu không đổi logic nên giữ nguyên)
    }
  }

  // ============================================================
  // window.clearLog: HÀM GLOBAL ĐỂ XOÁ LOG BOX
  // ============================================================
  // - Gắn vào window để có thể gọi từ HTML (onclick="clearLog()")
  // - Khi gọi:
  //   + reset logBox và show 1 dòng muted "Log đã được xoá."
  window.clearLog = function () {
    if (!logBox) return;
    logBox.innerHTML = '<div class="log__line log__line--muted">Log đã được xoá.</div>';
  };

  // ============================================================
  // GẮN EVENT CLICK CHO NÚT START
  // ============================================================
  // - Khi click btnStartCrawl -> gọi startJob()
  if (btn) btn.addEventListener("click", startJob);
})(); // Kết thúc IIFE: code chạy ngay khi file load
