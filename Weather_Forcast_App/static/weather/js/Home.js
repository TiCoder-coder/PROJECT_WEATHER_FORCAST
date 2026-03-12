console.log("✅ Home.js loaded"); // Log để xác nhận file JS đã được load (dễ debug trên DevTools Console)

document.addEventListener("DOMContentLoaded", () => {
  // ============================================================
  // KHỐI 1: HÀM TIỆN ÍCH MỞ / ĐÓNG MODAL (DÙNG CHUNG)
  // ============================================================
  // Ý tưởng:
  // - Modal được điều khiển bằng class CSS:
  //   + "is-open": hiển thị modal
  //   + "no-scroll": chặn scroll trang khi modal mở (gắn vào body)
  //
  // Tại sao cần no-scroll?
  // - Khi modal mở, user cuộn trang nền sẽ làm trải nghiệm xấu
  // - Nên thường sẽ disable scroll cho body
  const openModal = (modalEl) => {
    // Nếu element modal không tồn tại -> return (tránh lỗi null)
    if (!modalEl) return;

    // Thêm class "is-open" để CSS hiển thị modal
    modalEl.classList.add("is-open");

    // Thêm class "no-scroll" vào body để disable scroll trang nền
    document.body.classList.add("no-scroll");
  };

  const closeModal = (modalEl) => {
    // Nếu element modal không tồn tại -> return
    if (!modalEl) return;

    // Xóa class "is-open" để CSS ẩn modal
    modalEl.classList.remove("is-open");

    // Xóa class "no-scroll" để cho phép scroll lại
    document.body.classList.remove("no-scroll");
  };

  // ============================================================
  // KHỐI 2: MODAL CHỌN PHƯƠNG THỨC CRAWL (Crawl Method Modal)
  // ============================================================
  // Các element liên quan:
  // - btnOpenCrawl: nút bấm để mở modal chọn crawl
  // - crawlModal: modal chứa danh sách pipeline crawl
  // - chosen: element hiển thị "phương thức đã chọn" (label)
  const btnOpenCrawl = document.getElementById("btnOpenCrawlModal");
  const crawlModal = document.getElementById("crawlMethodModal");
  const chosen = document.getElementById("chosenCrawlMethod");

  // Chỉ setup event nếu cả nút và modal đều tồn tại trên trang
  if (btnOpenCrawl && crawlModal) {
    // Khi bấm nút => mở modal
    btnOpenCrawl.addEventListener("click", () => openModal(crawlModal));

    // Click bên trong modal: dùng event delegation để đóng modal khi click đúng phần tử có attribute data-close-crawl-modal
    // Lý do dùng delegation:
    // - Không cần add listener cho từng nút close
    // - Chỉ cần 1 listener trên container modal là đủ
    crawlModal.addEventListener("click", (e) => {
      const t = e.target; // phần tử thực sự được click
      // Nếu phần tử click có attribute data-close-crawl-modal -> đóng modal
      if (t && t.hasAttribute("data-close-crawl-modal")) {
        closeModal(crawlModal);
      }
    });

    // Tìm tất cả nút/pill có attribute data-crawl-method (mỗi nút là 1 pipeline)
    // Ví dụ HTML có thể như:
    // <button data-crawl-method="vrain_api" data-crawl-label="Vrain API">...</button>
    crawlModal.querySelectorAll("[data-crawl-method]").forEach((btn) => {
      btn.addEventListener("click", () => {
        // Lấy "method" dùng để quyết định route backend
        const method = btn.getAttribute("data-crawl-method");

        // Lấy label hiển thị cho user (nếu không có thì fallback = method)
        const label = btn.getAttribute("data-crawl-label") || method;

        // Nếu có element hiển thị lựa chọn -> update text cho đẹp UI
        if (chosen) chosen.textContent = label;

        // ============================================================
        // MAPPING METHOD -> ROUTE BACKEND
        // ============================================================
        // - Dựa vào method, chuyển hướng sang đúng URL xử lý crawl
        // - window.location.href sẽ điều hướng trang đến route mới
        // - Các route ví dụ:
        //   /crawl-api-weather/
        //   /crawl-vrain-html/
        //   /crawl-vrain-api/
        //   /crawl-vrain-selenium/
        //
        // Nếu method chưa được map -> alert để dev biết thiếu mapping
        if (method === "api_weather") {
          window.location.href = "/crawl-api-weather/";
        } else if (method === "vrain_html") {
          window.location.href = "/crawl-vrain-html/";
        } else if (method === "vrain_api") {
          window.location.href = "/crawl-vrain-api/";
        } else if (method === "vrain_selenium") {
          window.location.href = "/crawl-vrain-selenium/";
        } else {
          alert(`Pipeline "${label}" chưa map route backend.`);
        }

        // Sau khi chọn xong thì đóng modal (tránh modal còn hiện trên trang mới/hoặc UX gọn)
        closeModal(crawlModal);
      });
    });
  }

  // ============================================================
  // KHỐI 3: INTRO MODAL (giới thiệu)
  // ============================================================
  // - btnIntro: nút mở intro
  // - introModal: modal chứa nội dung giới thiệu/hướng dẫn
  const btnIntro = document.getElementById("btnIntro");
  const introModal = document.getElementById("introModal");

  if (btnIntro && introModal) {
    // Bấm nút -> mở intro modal
    btnIntro.addEventListener("click", () => openModal(introModal));

    // Click trong modal -> đóng nếu click đúng phần tử có data-close-intro-modal
    introModal.addEventListener("click", (e) => {
      const t = e.target;
      if (t && t.hasAttribute("data-close-intro-modal")) {
        closeModal(introModal);
      }
    });
  }

  // ============================================================
  // KHỐI 4: HELP MODAL (hướng dẫn / trợ giúp)
  // ============================================================
  // - btnHelp: nút mở help
  // - helpModal: modal trợ giúp
  const btnHelp = document.getElementById("btnOpenHelpModal");
  const helpModal = document.getElementById("helpModal");

  if (btnHelp && helpModal) {
    // Bấm nút -> mở help modal
    btnHelp.addEventListener("click", () => openModal(helpModal));

    // Click trong help modal -> đóng nếu click vào element có data-close-help-modal
    helpModal.addEventListener("click", (e) => {
      const t = e.target;
      if (t && t.hasAttribute("data-close-help-modal")) {
        closeModal(helpModal);
      }
    });
  }

  // ============================================================
  // KHỐI 5: ĐÓNG MODAL BẰNG PHÍM ESC (Keyboard UX)
  // ============================================================
  // - Khi user bấm Escape:
  //   + Nếu modal nào đang open -> đóng modal đó
  // - Kiểm tra classList.contains("is-open") để biết modal đang mở
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;

    if (crawlModal && crawlModal.classList.contains("is-open")) closeModal(crawlModal);
    if (introModal && introModal.classList.contains("is-open")) closeModal(introModal);
    if (helpModal && helpModal.classList.contains("is-open")) closeModal(helpModal);
  });

  // ============================================================
  // KHỐI 6: SCROLL ĐẾN SECTION DATASETS (smooth scroll)
  // ============================================================
  // - btnScroll: nút "xem datasets" hoặc "go to datasets"
  // - Khi click:
  //   + tìm element id="datasets"
  //   + scrollIntoView({behavior:"smooth"}) để cuộn mượt
  const btnScroll = document.getElementById("scroll-to-datasets");
  if (btnScroll) {
    btnScroll.addEventListener("click", () => {
      const el = document.getElementById("datasets");
      if (el) el.scrollIntoView({ behavior: "smooth" });
    });
  }
});

// ============================================================
// KHỐI 7: NÚT "START CRAWL" TRONG INTRO MODAL
// ============================================================
// Ý tưởng:
// - Trong intro modal có nút "Bắt đầu crawl" (introStartCrawl)
// - Khi bấm:
//   1) đóng intro modal (bằng cách giả lập click vào nút close trong intro)
//   2) mở crawl modal để user chọn phương thức crawl ngay
//
// Lưu ý:
// - Đây là một DOMContentLoaded khác, nghĩa là code sẽ chạy sau khi DOM ready.
// - Có thể gộp chung với listener ở trên, nhưng bạn yêu cầu không đổi logic nên giữ nguyên.
document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.getElementById("introStartCrawl");
  if (!startBtn) return;

  startBtn.addEventListener("click", () => {
    // Tìm nút close trong intro modal theo selector:
    // "#introModal [data-close-intro-modal]"
    // -> giả lập click để dùng cùng logic đóng modal đang có
    const close = document.querySelector("#introModal [data-close-intro-modal]");
    if (close) close.click();

    // Mở modal crawl bằng cách giả lập click vào nút open crawl modal
    const openCrawl = document.getElementById("btnOpenCrawlModal");
    if (openCrawl) openCrawl.click();
  });
});

// ============================================================
// KHỐI 8: QUICK CRAWL BUTTON (CHO USER ĐÃ LOGIN)
// ============================================================
// Ý tưởng:
// - Có một nút "Quick Crawl" hiển thị cho người dùng đã đăng nhập
// - Khi bấm:
//   + mở luôn crawl modal (giống như bấm nút chọn crawl)
// - Giúp thao tác nhanh hơn, không cần vào intro hoặc tìm nút khác
document.addEventListener("DOMContentLoaded", () => {
  const quickCrawlBtn = document.getElementById("btnQuickCrawl");
  if (!quickCrawlBtn) return;

  quickCrawlBtn.addEventListener("click", () => {
    const openCrawl = document.getElementById("btnOpenCrawlModal");
    if (openCrawl) openCrawl.click();
  });
});
 