/**
 * JS_Predict.js – Frontend logic cho trang dự báo thời tiết
 * ==========================================================
 * - Tab switching (Dataset / Upload / Manual)
 * - Folder → File dropdown
 * - "Dùng" dataset button
 * - Dataset form submit (fetch FormData POST)
 * - Upload form submit (fetch FormData POST)
 * - Manual form submit (fetch JSON POST)
 * - Log polling (tail) + preview rendering
 * - Auto-resume on page load
 */

document.addEventListener('DOMContentLoaded', () => {
  /* ── Data from Django ── */
  const CSRF = document.querySelector('meta[name="csrf-token"]')?.content || '';
  const ALL_DATASETS = JSON.parse(
    document.getElementById('datasets-data')?.textContent || '[]'
  );
  const activeJobMeta = document.querySelector('meta[name="active-job-id"]');
  let currentJobId = (activeJobMeta && activeJobMeta.content) ? activeJobMeta.content : null;
  let pollTimer = null;
  let logAfter = 0;

  /* ── DOM refs ── */
  const $folderKey   = document.getElementById('folderKey');
  const $fileName    = document.getElementById('fileName');

  const $datasetForm = document.getElementById('datasetForm');
  const $uploadForm  = document.getElementById('uploadForm');
  const $manualForm  = document.getElementById('manualForm');

  const $fileInput   = document.getElementById('fileInput');
  const $uploadZone  = document.getElementById('uploadZone');
  const $selectedFile = document.getElementById('selectedFile');

  const $btnRunDataset = document.getElementById('btnRunDataset');
  const $btnRunUpload  = document.getElementById('btnRunUpload');
  const $btnRunManual  = document.getElementById('btnRunManual');
  const $btnResetManual = document.getElementById('btnResetManual');
  const $btnForecastNow = document.getElementById('btnForecastNow');
  const $btnRefresh    = document.getElementById('btnRefreshDatasets');
  const $btnToggleRecent = document.getElementById('btnToggleRecent');

  const $sectionLogs  = document.getElementById('sectionLogs');
  const $logPre       = document.getElementById('logPre');
  const $progressFill = document.getElementById('progressFill');
  const $progressPct  = document.getElementById('progressPercent');
  const $progressStep = document.getElementById('progressStep');
  const $predictStatus = document.getElementById('predictStatus');

  const $sectionResults = document.getElementById('sectionResults');
  const $statsBadge    = document.getElementById('statsBadge');
  const $statsGrid     = document.getElementById('statsGrid');
  const $resultThead   = document.getElementById('resultThead');
  const $resultTbody   = document.getElementById('resultTbody');

  const $manualResult  = document.getElementById('manualResult');
  const $manualResultBody = document.getElementById('manualResultBody');

  // ═══════════════════════════════════════════════
  // Tab switching
  // ═══════════════════════════════════════════════
  document.querySelectorAll('.predict-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.predict-tab').forEach(t => t.classList.remove('predict-tab--active'));
      document.querySelectorAll('.predict-panel').forEach(p => p.classList.remove('predict-panel--active'));

      tab.classList.add('predict-tab--active');
      const panel = document.querySelector(`.predict-panel[data-panel="${tab.dataset.tab}"]`);
      if (panel) panel.classList.add('predict-panel--active');
    });
  });

  // ═══════════════════════════════════════════════
  // Folder → File dropdown
  // ═══════════════════════════════════════════════
  function updateFileDropdown(folderKey) {
    $fileName.innerHTML = '<option value="">-- Chọn file --</option>';
    if (!folderKey) return;

    const files = ALL_DATASETS.filter(d => d.folder_key === folderKey);
    files.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f.filename;
      opt.textContent = `${f.filename}  (${f.size_mb} MB)`;
      $fileName.appendChild(opt);
    });
  }

  if ($folderKey) {
    $folderKey.addEventListener('change', () => updateFileDropdown($folderKey.value));
  }

  // ═══════════════════════════════════════════════
  // "Dùng" dataset buttons
  // ═══════════════════════════════════════════════
  function attachUseButtons(scope) {
    (scope || document).querySelectorAll('.btn-use-dataset').forEach(btn => {
      btn.addEventListener('click', () => {
        const folder = btn.dataset.folder;
        const file   = btn.dataset.file;

        // Switch to dataset tab
        document.querySelectorAll('.predict-tab').forEach(t => t.classList.remove('predict-tab--active'));
        document.querySelectorAll('.predict-panel').forEach(p => p.classList.remove('predict-panel--active'));
        const dsTab = document.querySelector('.predict-tab[data-tab="dataset"]');
        const dsPanel = document.querySelector('.predict-panel[data-panel="dataset"]');
        if (dsTab) dsTab.classList.add('predict-tab--active');
        if (dsPanel) dsPanel.classList.add('predict-panel--active');

        // Set values
        if ($folderKey) {
          $folderKey.value = folder;
          updateFileDropdown(folder);
          setTimeout(() => { $fileName.value = file; }, 50);
        }

        // Scroll to predict section
        document.getElementById('sectionPredict')?.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Visual feedback
        btn.textContent = '✔ Đã chọn';
        btn.style.opacity = '0.6';
        setTimeout(() => { btn.textContent = '🔮 Dùng'; btn.style.opacity = '1'; }, 1200);
      });
    });
  }
  attachUseButtons();

  // ═══════════════════════════════════════════════
  // File upload zone
  // ═══════════════════════════════════════════════
  if ($fileInput) {
    $fileInput.addEventListener('change', () => {
      if ($fileInput.files.length > 0) {
        const f = $fileInput.files[0];
        $selectedFile.style.display = '';
        $selectedFile.textContent = `📄 ${f.name}  (${(f.size / 1024 / 1024).toFixed(2)} MB)`;
      }
    });
  }

  if ($uploadZone) {
    $uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      $uploadZone.classList.add('upload-zone--dragover');
    });
    $uploadZone.addEventListener('dragleave', () => {
      $uploadZone.classList.remove('upload-zone--dragover');
    });
    $uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      $uploadZone.classList.remove('upload-zone--dragover');
      if (e.dataTransfer.files.length > 0) {
        $fileInput.files = e.dataTransfer.files;
        $fileInput.dispatchEvent(new Event('change'));
      }
    });
  }

  // ═══════════════════════════════════════════════
  // Dataset form submit
  // ═══════════════════════════════════════════════
  if ($datasetForm) {
    $datasetForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      const folderKey = $folderKey?.value || '';
      const filename  = $fileName?.value || '';

      if (!folderKey) { alert('⚠️ Vui lòng chọn thư mục dữ liệu'); return; }
      if (!filename)  { alert('⚠️ Vui lòng chọn file dữ liệu'); return; }

      const formData = new FormData();
      formData.append('folder_key', folderKey);
      formData.append('filename', filename);
      const nrows = document.getElementById('nrowsDataset')?.value || '0';
      formData.append('nrows', nrows);

      await submitPrediction(formData, $btnRunDataset);
    });
  }

  // ═══════════════════════════════════════════════
  // Upload form submit
  // ═══════════════════════════════════════════════
  if ($uploadForm) {
    $uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      if (!$fileInput || !$fileInput.files.length) {
        alert('⚠️ Vui lòng chọn file để upload');
        return;
      }

      const formData = new FormData();
      formData.append('file', $fileInput.files[0]);
      const nrows = document.getElementById('nrowsUpload')?.value || '0';
      formData.append('nrows', nrows);

      await submitPrediction(formData, $btnRunUpload);
    });
  }

  // ═══════════════════════════════════════════════
  // Generic prediction submit
  // ═══════════════════════════════════════════════
  async function submitPrediction(formData, submitBtn) {
    return submitPredictionToUrl('/predict/run/', formData, submitBtn);
  }

  async function submitPredictionToUrl(url, formDataOrBody, submitBtn, isJson = false) {
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = '⏳ Đang khởi chạy...';
    }

    try {
      const headers = { 'X-CSRFToken': CSRF };
      let body = formDataOrBody;
      if (isJson) {
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify(formDataOrBody);
      }

      const resp = await fetch(url, {
        method: 'POST',
        headers,
        body,
      });

      const data = await resp.json();

      if (!data.ok) {
        alert('❌ ' + (data.error || 'Lỗi không xác định'));
        resetButton(submitBtn);
        return;
      }

      currentJobId = data.job_id;
      showLogs();
      startPolling();
    } catch (err) {
      alert('❌ Network error: ' + err.message);
      resetButton(submitBtn);
    }
  }

  function resetButton(btn) {
    if (!btn) return;
    btn.disabled = false;
    btn.textContent = '🔮 Dự báo';
  }

  function resetAllButtons() {
    resetButton($btnRunDataset);
    resetButton($btnRunUpload);
    if ($btnForecastNow) {
      $btnForecastNow.disabled = false;
      $btnForecastNow.textContent = '⚡ Dự báo ngay';
    }
  }

  // ═══════════════════════════════════════════════
  // Forecast Now button
  // ═══════════════════════════════════════════════
  if ($btnForecastNow) {
    $btnForecastNow.addEventListener('click', async () => {
      await submitPredictionToUrl('/predict/forecast-now/', {}, $btnForecastNow, true);
    });
  }

  // ═══════════════════════════════════════════════
  // Manual form submit
  // ═══════════════════════════════════════════════
  if ($manualForm) {
    $manualForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      // Collect form data into a row object
      const formData = new FormData($manualForm);
      const row = {};
      for (const [key, val] of formData.entries()) {
        if (key === 'csrfmiddlewaretoken') continue;
        row[key] = val;
      }

      // Add defaults for missing columns
      if (!row.station_id) row.station_id = '99';
      if (!row.latitude) row.latitude = '21.0285';
      if (!row.longitude) row.longitude = '105.8542';
      if (!row.timestamp) row.timestamp = new Date().toISOString().split('T')[0];
      if (!row.data_source) row.data_source = 'manual';
      if (!row.data_quality) row.data_quality = 'manual';
      if (!row.data_time) row.data_time = new Date().toISOString();

      // Compute averages from what we have
      if (row.temperature_current && row.temperature_max && row.temperature_min) {
        row.temperature_avg = ((parseFloat(row.temperature_current) + parseFloat(row.temperature_max) + parseFloat(row.temperature_min)) / 3).toFixed(2);
      }
      if (row.humidity_current && row.humidity_max && row.humidity_min) {
        row.humidity_avg = ((parseFloat(row.humidity_current) + parseFloat(row.humidity_max) + parseFloat(row.humidity_min)) / 3).toFixed(2);
      }

      if ($btnRunManual) {
        $btnRunManual.disabled = true;
        $btnRunManual.textContent = '⏳ Đang dự báo...';
      }

      try {
        const resp = await fetch('/predict/manual/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': CSRF,
          },
          body: JSON.stringify({ rows: [row] }),
        });

        const data = await resp.json();

        if (!data.ok) {
          alert('❌ ' + (data.error || 'Lỗi không xác định'));
          if ($btnRunManual) {
            $btnRunManual.disabled = false;
            $btnRunManual.textContent = '🔮 Dự báo ngay';
          }
          return;
        }

        // Show result
        if ($manualResult && $manualResultBody && data.predictions && data.predictions.length > 0) {
          const pred = data.predictions[0];
          const val = pred.y_pred;
          const status = pred.rain_status;

          let html = `
            <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
              <div class="manual-result__pred-value">${val}</div>
              <div>
                <div style="font-size:16px; font-weight:700; margin-bottom:4px;">
                  ${status === 'Mưa' ? '🌧️' : '☀️'} ${status}
                </div>
                <div style="font-size:13px; color:#9ca3af;">
                  Trạm: ${pred.station_name || row.station_name || '—'} · 
                  Tỉnh: ${pred.province || row.province || '—'}
                </div>
                <div style="font-size:12px; color:#64748b; margin-top:4px;">
                  ⏱️ Thời gian dự báo: ${data.stats?.prediction_time || '—'}s
                </div>
              </div>
            </div>
          `;
          $manualResultBody.innerHTML = html;
          $manualResult.style.display = '';
          $manualResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

      } catch (err) {
        alert('❌ Network error: ' + err.message);
      }

      if ($btnRunManual) {
        $btnRunManual.disabled = false;
        $btnRunManual.textContent = '🔮 Dự báo ngay';
      }
    });
  }

  // ═══════════════════════════════════════════════
  // Log panel helpers
  // ═══════════════════════════════════════════════
  function showLogs() {
    if ($sectionLogs) $sectionLogs.style.display = '';
    if ($logPre) $logPre.textContent = '';
    if ($predictStatus) {
      $predictStatus.textContent = '⏳ Đang dự báo...';
      $predictStatus.className = 'predict-status';
    }
    if ($sectionResults) $sectionResults.style.display = 'none';
    $sectionLogs?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function startPolling() {
    logAfter = 0;
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollLogs, 1500);
    pollLogs(); // immediately
  }

  async function pollLogs() {
    if (!currentJobId) return;

    try {
      const resp = await fetch(`/predict/tail/?job_id=${currentJobId}&after=${logAfter}`);
      const data = await resp.json();

      if (!data.ok) return;

      // Append logs
      if (data.logs && data.logs.length > 0) {
        data.logs.forEach(line => {
          $logPre.textContent += line + '\n';
        });
        logAfter = data.total;

        // Auto-scroll
        const logOutput = document.getElementById('logOutput');
        if (logOutput) logOutput.scrollTop = logOutput.scrollHeight;
      }

      // Update progress
      if ($progressFill)  $progressFill.style.width = data.progress + '%';
      if ($progressPct)   $progressPct.textContent = data.progress + '%';
      if ($progressStep)  $progressStep.textContent = data.step || '';

      // Check completion
      if (data.status === 'done') {
        clearInterval(pollTimer);
        if ($predictStatus) {
          $predictStatus.textContent = '✅ Hoàn tất!';
          $predictStatus.className = 'predict-status predict-status--done';
        }
        resetAllButtons();

        // Render results
        if (data.preview && data.preview.length > 0) {
          renderResults(data);
        }

      } else if (data.status === 'error') {
        clearInterval(pollTimer);
        if ($predictStatus) {
          $predictStatus.textContent = '❌ Lỗi!';
          $predictStatus.className = 'predict-status predict-status--error';
        }
        resetAllButtons();
      }
    } catch (err) {
      console.error('Poll error:', err);
    }
  }

  // ═══════════════════════════════════════════════
  // Render results table + stats
  // ═══════════════════════════════════════════════
  function renderResults(data) {
    if (!$sectionResults) return;
    $sectionResults.style.display = '';

    // Stats badge
    if ($statsBadge && data.stats) {
      $statsBadge.textContent = `${data.stats.n_samples} mẫu · ${data.stats.prediction_time}s`;
    }

    // Stats grid
    if ($statsGrid && data.stats) {
      const s = data.stats;
      $statsGrid.innerHTML = `
        <div class="stat-card">
          <div class="stat-card__label">Số mẫu</div>
          <div class="stat-card__value">${s.n_samples}</div>
        </div>
        <div class="stat-card stat-card--accent">
          <div class="stat-card__label">Trung bình</div>
          <div class="stat-card__value">${s.mean}</div>
        </div>
        <div class="stat-card stat-card--warn">
          <div class="stat-card__label">Std</div>
          <div class="stat-card__value">${s.std}</div>
        </div>
        <div class="stat-card stat-card--good">
          <div class="stat-card__label">Min</div>
          <div class="stat-card__value">${s.min}</div>
        </div>
        <div class="stat-card">
          <div class="stat-card__label">Max</div>
          <div class="stat-card__value">${s.max}</div>
        </div>
        <div class="stat-card stat-card--accent">
          <div class="stat-card__label">Thời gian</div>
          <div class="stat-card__value">${s.prediction_time}s</div>
        </div>
      `;
    }

    // Table — map raw column names to user-friendly Vietnamese headers
    const colHeaders = {
      'station_name': 'Trạm',
      'province': 'Tỉnh/TP',
      'district': 'Quận/Huyện',
      'rain_total': 'Mưa thực tế',
      'status': 'Trạng thái',
      'y_pred': 'Dự báo (y_pred)',
      'forecast_for': 'Thời gian dự báo',
      'data_collected_at': 'Thu thập lúc',
    };
    const columns = data.preview_columns || [];
    const rows = data.preview || [];

    if ($resultThead && columns.length > 0) {
      $resultThead.innerHTML = '<tr>' + columns.map(c =>
        `<th>${escHtml(colHeaders[c] || c)}</th>`
      ).join('') + '</tr>';
    }

    if ($resultTbody && rows.length > 0) {
      $resultTbody.innerHTML = rows.map(row =>
        '<tr>' + columns.map(c => {
          const val = row[c] !== undefined ? row[c] : '';
          const cls = c === 'y_pred' ? ' class="td-pred"'
            : c === 'status' ? ` class="${val === 'Mưa' ? 'td-rain' : 'td-ok'}"`
            : '';
          if (c === 'status') {
            const badge = val === 'Mưa' ? 'badge--rain' : 'badge--ok';
            return `<td><span class="badge ${badge}">${escHtml(String(val))}</span></td>`;
          }
          return `<td${cls}>${escHtml(String(val))}</td>`;
        }).join('') + '</tr>'
      ).join('');
    }

    $sectionResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function escHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ═══════════════════════════════════════════════
  // Guide toggle
  // ═══════════════════════════════════════════════
  const $btnToggleGuide = document.getElementById('btnToggleGuide');
  const $guideBody = document.getElementById('guideBody');
  if ($btnToggleGuide && $guideBody) {
    $btnToggleGuide.addEventListener('click', () => {
      const collapsed = $guideBody.classList.toggle('guide-body--collapsed');
      $btnToggleGuide.textContent = collapsed ? 'Mở rộng' : 'Thu gọn';
    });
  }

  // ═══════════════════════════════════════════════
  // Recent predictions toggle
  // ═══════════════════════════════════════════════
  if ($btnToggleRecent) {
    const $recentBody = document.getElementById('recentBody');
    $btnToggleRecent.addEventListener('click', () => {
      if (!$recentBody) return;
      const collapsed = $recentBody.classList.toggle('recent-body--collapsed');
      $btnToggleRecent.textContent = collapsed ? 'Mở rộng' : 'Thu gọn';
    });
  }

  // ═══════════════════════════════════════════════
  // Refresh datasets
  // ═══════════════════════════════════════════════
  if ($btnRefresh) {
    $btnRefresh.addEventListener('click', async () => {
      $btnRefresh.disabled = true;
      $btnRefresh.textContent = '⏳...';

      try {
        const resp = await fetch('/predict/model-info/');
        const data = await resp.json();

        if (data.ok) {
          // Just reload the page for now — simplest way to refresh datasets
          location.reload();
          return;
        }
      } catch (err) {
        console.error('Refresh error:', err);
      }

      $btnRefresh.disabled = false;
      $btnRefresh.textContent = '🔄 Refresh';
    });
  }

  // ═══════════════════════════════════════════════
  // Auto-resume polling if there's an active job
  // ═══════════════════════════════════════════════
  if (currentJobId) {
    showLogs();
    startPolling();
  }
});
