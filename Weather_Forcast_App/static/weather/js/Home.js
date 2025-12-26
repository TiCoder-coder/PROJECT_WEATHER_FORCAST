console.log("✅ Home.js loaded");

document.addEventListener("DOMContentLoaded", () => {
  const btnOpen = document.getElementById("btnOpenCrawlModal");
  const modal = document.getElementById("crawlMethodModal");
  const chosen = document.getElementById("chosenCrawlMethod");

  if (!btnOpen || !modal) return;

  const openModal = () => modal.classList.add("is-open");
  const closeModal = () => modal.classList.remove("is-open");

  btnOpen.addEventListener("click", openModal);

  modal.addEventListener("click", (e) => {
    const target = e.target;
    if (target && target.hasAttribute("data-close-crawl-modal")) {
      closeModal();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });

  modal.querySelectorAll("[data-crawl-method]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const method = btn.getAttribute("data-crawl-method");
      const label = btn.getAttribute("data-crawl-label") || method;

      if (chosen) chosen.textContent = label;
      if (method === "api_weather") {
        window.location.href = "/crawl-api-weather/";
      } else if (method === "vrain_html") {
        window.location.href = "/crawl-vrain-html/";
      } else {
        alert(`Pipeline "${label}" chưa map route backend. Tạm thời chỉ mở được API Weather & Vrain HTML.`);
      }

      closeModal();
    });
  });

  const btnScroll = document.getElementById("scroll-to-datasets");
  if (btnScroll) {
    btnScroll.addEventListener("click", () => {
      const el = document.getElementById("datasets");
      if (el) el.scrollIntoView({ behavior: "smooth" });
    });
  }
});
