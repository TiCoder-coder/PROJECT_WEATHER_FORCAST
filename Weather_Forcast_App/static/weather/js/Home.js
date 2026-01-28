console.log("✅ Home.js loaded");

document.addEventListener("DOMContentLoaded", () => {
  const openModal = (modalEl) => {
    if (!modalEl) return;
    modalEl.classList.add("is-open");
    document.body.classList.add("no-scroll");
  };

  const closeModal = (modalEl) => {
    if (!modalEl) return;
    modalEl.classList.remove("is-open");
    document.body.classList.remove("no-scroll");
  };

  const btnOpenCrawl = document.getElementById("btnOpenCrawlModal");
  const crawlModal = document.getElementById("crawlMethodModal");
  const chosen = document.getElementById("chosenCrawlMethod");

  if (btnOpenCrawl && crawlModal) {
    btnOpenCrawl.addEventListener("click", () => openModal(crawlModal));

    crawlModal.addEventListener("click", (e) => {
      const t = e.target;
      if (t && t.hasAttribute("data-close-crawl-modal")) {
        closeModal(crawlModal);
      }
    });

    crawlModal.querySelectorAll("[data-crawl-method]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const method = btn.getAttribute("data-crawl-method");
        const label = btn.getAttribute("data-crawl-label") || method;

        if (chosen) chosen.textContent = label;

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

        closeModal(crawlModal);
      });
    });
  }

  const btnIntro = document.getElementById("btnIntro");
  const introModal = document.getElementById("introModal");

  if (btnIntro && introModal) {
    btnIntro.addEventListener("click", () => openModal(introModal));

    introModal.addEventListener("click", (e) => {
      const t = e.target;
      if (t && t.hasAttribute("data-close-intro-modal")) {
        closeModal(introModal);
      }
    });
  }

  const btnHelp = document.getElementById("btnOpenHelpModal");
  const helpModal = document.getElementById("helpModal");

  if (btnHelp && helpModal) {
    btnHelp.addEventListener("click", () => openModal(helpModal));

    helpModal.addEventListener("click", (e) => {
      const t = e.target;
      if (t && t.hasAttribute("data-close-help-modal")) {
        closeModal(helpModal);
      }
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;

    if (crawlModal && crawlModal.classList.contains("is-open")) closeModal(crawlModal);
    if (introModal && introModal.classList.contains("is-open")) closeModal(introModal);
    if (helpModal && helpModal.classList.contains("is-open")) closeModal(helpModal);
  });

  const btnScroll = document.getElementById("scroll-to-datasets");
  if (btnScroll) {
    btnScroll.addEventListener("click", () => {
      const el = document.getElementById("datasets");
      if (el) el.scrollIntoView({ behavior: "smooth" });
    });
  }
});
document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.getElementById("introStartCrawl");
  if (!startBtn) return;

  startBtn.addEventListener("click", () => {
    const close = document.querySelector("#introModal [data-close-intro-modal]");
    if (close) close.click();

    const openCrawl = document.getElementById("btnOpenCrawlModal");
    if (openCrawl) openCrawl.click();
  });
});

// Quick Crawl button for logged-in users
document.addEventListener("DOMContentLoaded", () => {
  const quickCrawlBtn = document.getElementById("btnQuickCrawl");
  if (!quickCrawlBtn) return;

  quickCrawlBtn.addEventListener("click", () => {
    const openCrawl = document.getElementById("btnOpenCrawlModal");
    if (openCrawl) openCrawl.click();
  });
});