console.log("✅ JS_Crawl_data_by_API.js loaded");

document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form.form");
  const logContainer = document.getElementById("log-container");
  const startBtn = document.querySelector('button[type="submit"][name="action"][value="start"]');

  const START_URL = "/crawl-api-weather/";
  const LOGS_URL = "/crawl-api-weather/logs/";

  let pollTimer = null;

  function setLog(lines) {
    if (!logContainer) return;
    logContainer.innerHTML = "";
    (lines || []).forEach((line) => {
      const div = document.createElement("div");
      div.className = "log__line";
      div.textContent = line;
      logContainer.appendChild(div);
    });
    logContainer.scrollTop = logContainer.scrollHeight;
  }

  async function fetchLogs() {
    const res = await fetch(LOGS_URL, {
      headers: { "X-Requested-With": "XMLHttpRequest" },
      cache: "no-store",
      credentials: "same-origin",
    });

    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      const text = await res.text();
      throw new Error("Logs endpoint không trả JSON. Response: " + text.slice(0, 120));
    }

    return await res.json();
  }

  async function startJob(formData) {
    const res = await fetch(START_URL, {
      method: "POST",
      body: formData,
      headers: { "X-Requested-With": "XMLHttpRequest" },
      credentials: "same-origin",
    });

    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      const text = await res.text();
      throw new Error("Start endpoint không trả JSON. Response: " + text.slice(0, 120));
    }

    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "start failed");
    return data;
  }

  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    formData.set("action", "start");

    if (startBtn) {
      startBtn.disabled = true;
      startBtn.textContent = "⏳ Đang chạy... vui lòng chờ";
    }

    if (pollTimer) clearInterval(pollTimer);

    try {
      await startJob(formData);

      const first = await fetchLogs();
      const sizeEl = document.getElementById("lastFileSize");
      if (sizeEl && d.csv_size_mb != null) sizeEl.textContent = `${d.csv_size_mb} MB`;

      setLog(first.logs);

      pollTimer = setInterval(async () => {
        try {
          const d = await fetchLogs();
          setLog(d.logs);

          if (!d.is_running) {
            clearInterval(pollTimer);
            pollTimer = null;
            if (startBtn) {
              startBtn.disabled = false;
              startBtn.textContent = "🚀 Bắt đầu crawl ngay";
            }
          }
        } catch (err) {
          setLog([`[ERROR] ${err.message}`]);
        }
      }, 1000);
    } catch (err) {
      setLog([`[ERROR] ${err.message}`]);
      if (startBtn) {
        startBtn.disabled = false;
        startBtn.textContent = "🚀 Bắt đầu crawl ngay";
      }
    }
  });
});
