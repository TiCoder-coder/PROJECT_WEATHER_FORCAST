(function () {
  const cfg = window.__VRAIN_API__ || {};
  const logBox = document.getElementById("log-container");
  const btn = document.getElementById("btnStartCrawl");
  const spinner = document.getElementById("spinner");
  const statusValue = document.getElementById("statusValue");
  const lastCrawlTime = document.getElementById("lastCrawlTime");
  const lastFileSize = document.getElementById("lastFileSize");

  let since = 0;
  let timer = null;

  function getCookie(name) {
    const parts = document.cookie ? document.cookie.split("; ") : [];
    for (const part of parts) {
      const [k, v] = part.split("=");
      if (k === name) return decodeURIComponent(v || "");
    }
    return "";
  }

  function setRunningUI(isRunning) {
    if (spinner) spinner.style.display = isRunning ? "inline-block" : "none";
    if (statusValue) statusValue.textContent = isRunning ? "🔄 Đang chạy..." : "✅ Sẵn sàng";
    if (btn) btn.disabled = isRunning;
  }

  function appendLines(lines) {
    if (!logBox || !lines || lines.length === 0) return;

    const muted = logBox.querySelector(".log__line--muted");
    if (muted) muted.remove();

    for (const line of lines) {
      const div = document.createElement("div");
      div.className = "log__line";
      div.textContent = line;
      logBox.appendChild(div);
    }
    logBox.scrollTop = logBox.scrollHeight;
  }

  async function startJob() {
    if (!cfg.startUrl) {
      alert("Thiếu startUrl.");
      return;
    }
    setRunningUI(true);

    try {
      const csrf = cfg.csrfToken || getCookie("csrftoken");
      const res = await fetch(cfg.startUrl, {
        method: "POST",
        credentials: "same-origin",
        headers: {
          "X-CSRFToken": csrf,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || ("HTTP " + res.status));
      }

      since = 0;
      if (logBox) {
        logBox.innerHTML =
          '<div class="log__line log__line--muted">Đang chạy… log sẽ cập nhật realtime.</div>';
      }

      if (timer) clearInterval(timer);
      timer = setInterval(pollLogs, 900);
      await pollLogs();
    } catch (e) {
      setRunningUI(false);
      alert("Start crawl lỗi: " + (e?.message || e));
    }
  }

  async function pollLogs() {
    if (!cfg.tailUrl) return;

    try {
      const url = new URL(cfg.tailUrl, window.location.origin);
      url.searchParams.set("since", String(since));

      const res = await fetch(url.toString(), { method: "GET", credentials: "same-origin" });
      if (!res.ok) return;

      const data = await res.json();
      if (!data.ok) return;

      appendLines(data.lines || []);
      since = data.next_since ?? since;

      if (data.last_crawl_time && lastCrawlTime) lastCrawlTime.textContent = data.last_crawl_time;
      if (typeof data.last_size_mb !== "undefined" && lastFileSize) {
        lastFileSize.textContent = data.last_size_mb ? `${data.last_size_mb} MB` : "–";
      }

      setRunningUI(!!data.is_running);

      if (!data.is_running && timer) {
        clearInterval(timer);
        timer = null;
      }
    } catch (e) {
    }
  }

  window.clearLog = function () {
    if (!logBox) return;
    logBox.innerHTML = '<div class="log__line log__line--muted">Log đã được xoá.</div>';
  };

  if (btn) btn.addEventListener("click", startJob);
})();
