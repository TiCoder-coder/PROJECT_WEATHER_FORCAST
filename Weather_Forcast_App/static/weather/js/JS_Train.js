/**
 * JS_Train.js – Frontend logic cho trang huấn luyện ML
 * =====================================================
 * - Tab switching (Quick / Advanced config)
 * - Dataset folder → file dropdown
 * - "Dùng" dataset button
 * - Submit training form (fetch POST)
 * - Log polling (tail)
 * - Artifacts toggle
 */

document.addEventListener('DOMContentLoaded', () => {
  // Read data from Django json_script and meta tags (no inline script needed)
  const CSRF = document.querySelector('meta[name="csrf-token"]')?.content || '';
  const ALL_DATASETS = JSON.parse(
    document.getElementById('datasets-data')?.textContent || '[]'
  );
  let pollTimer = null;
  const activeJobMeta = document.querySelector('meta[name="active-job-id"]');
  let currentJobId = (activeJobMeta && activeJobMeta.content) ? activeJobMeta.content : null;

  // ═══════════════════════════════════════════════
  // DOM refs
  // ═══════════════════════════════════════════════
  const $folderKey           = document.getElementById('folderKey');
  const $fileName            = document.getElementById('fileName');
  const $modelType           = document.getElementById('modelType'); // hidden select (kept for compat)
  const $modelTypeSelect     = document.getElementById('modelTypeSelect'); // visible dropdown
  const $stackingParamsPanel = document.getElementById('stackingParamsPanel');
  const $targetColumn        = document.getElementById('targetColumn');

  // ─── Model dropdown helper ──────────────────────────────────
  function getSelectedModelType() {
    return $modelTypeSelect ? $modelTypeSelect.value : 'ensemble';
  }

  // Keep hidden select in sync with dropdown (for forms that still look at #modelType)
  function syncModelTypeSelect(val) {
    if ($modelType) $modelType.value = val;
  }

  // Show / hide stacking params panel
  function updateStackingPanel() {
    const v = getSelectedModelType();
    if ($stackingParamsPanel) {
      $stackingParamsPanel.style.display = (v === 'stacking_ensemble') ? '' : 'none';
    }
  }

  // Attach change handler to the model dropdown
  if ($modelTypeSelect) {
    $modelTypeSelect.addEventListener('change', () => {
      const v = getSelectedModelType();
      syncModelTypeSelect(v);
      updateStackingPanel();
    });
  }

  // Initial state
  updateStackingPanel();
  syncModelTypeSelect(getSelectedModelType());
  const $testSize      = document.getElementById('testSize');
  const $validSize     = document.getElementById('validSize');
  const $useDefault    = document.getElementById('useDefaultConfig');
  const $configJson    = document.getElementById('configJson');

  const $trainForm     = document.getElementById('trainForm');
  const $btnStart      = document.getElementById('btnStartTrain');
  const $btnReset      = document.getElementById('btnResetForm');
  const $btnRefresh    = document.getElementById('btnRefreshDatasets');

  const $sectionLogs   = document.getElementById('sectionLogs');
  const $logPre        = document.getElementById('logPre');
  const $progressFill  = document.getElementById('progressFill');
  const $progressPct   = document.getElementById('progressPercent');
  const $progressStep  = document.getElementById('progressStep');
  const $trainStatus   = document.getElementById('trainStatus');

  const $btnToggle     = document.getElementById('btnToggleArtifacts');
  const $artifactsBody = document.getElementById('artifactsBody');

  // ═══════════════════════════════════════════════
  // Tab switching
  // ═══════════════════════════════════════════════
  document.querySelectorAll('.config-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.config-tab').forEach(t => t.classList.remove('config-tab--active'));
      document.querySelectorAll('.config-panel').forEach(p => p.classList.remove('config-panel--active'));

      tab.classList.add('config-tab--active');
      const panel = document.querySelector(`.config-panel[data-panel="${tab.dataset.tab}"]`);
      if (panel) panel.classList.add('config-panel--active');
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
  document.querySelectorAll('.btn-use-dataset').forEach(btn => {
    btn.addEventListener('click', () => {
      const folder = btn.dataset.folder;
      const file   = btn.dataset.file;

      if ($folderKey) {
        $folderKey.value = folder;
        updateFileDropdown(folder);
        setTimeout(() => { $fileName.value = file; }, 50);
      }

      // Scroll to config
      document.getElementById('sectionConfig')?.scrollIntoView({ behavior: 'smooth', block: 'start' });

      // Visual feedback
      btn.textContent = '✔ Đã chọn';
      btn.style.opacity = '0.6';
      setTimeout(() => { btn.textContent = '✅ Dùng'; btn.style.opacity = '1'; }, 1200);
    });
  });

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
  // Artifacts toggle
  // ═══════════════════════════════════════════════
  if ($btnToggle && $artifactsBody) {
    $btnToggle.addEventListener('click', () => {
      const hidden = $artifactsBody.style.display === 'none';
      $artifactsBody.style.display = hidden ? '' : 'none';
      $btnToggle.textContent = hidden ? 'Thu gọn' : 'Mở rộng';
    });
  }

  // Stacking artifacts toggle
  const $btnToggleStacking = document.getElementById('btnToggleStackingArtifacts');
  const $stackingArtifactsBody = document.getElementById('stackingArtifactsBody');
  if ($btnToggleStacking && $stackingArtifactsBody) {
    $btnToggleStacking.addEventListener('click', () => {
      const hidden = $stackingArtifactsBody.style.display === 'none';
      $stackingArtifactsBody.style.display = hidden ? '' : 'none';
      $btnToggleStacking.textContent = hidden ? 'Thu gọn' : 'Mở rộng';
    });
  }

  // BestParams toggle
  const $btnToggleBestParams = document.getElementById('btnToggleBestParams');
  const $bestParamsBody = document.getElementById('bestParamsBody');
  if ($btnToggleBestParams && $bestParamsBody) {
    $btnToggleBestParams.addEventListener('click', () => {
      const hidden = $bestParamsBody.style.display === 'none';
      $bestParamsBody.style.display = hidden ? '' : 'none';
      $btnToggleBestParams.textContent = hidden ? 'Thu gọn' : 'Mở rộng';
    });
  }

  // ═══════════════════════════════════════════════
  // Submit train
  // ═══════════════════════════════════════════════
  if ($trainForm) {
    $trainForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      // Check if advanced tab is active
      const advancedActive = document.querySelector('.config-tab--active')?.dataset.tab === 'advanced';

      let body;

      if (advancedActive && $configJson) {
        // Parse JSON textarea — extract folder_key/filename for backend validation
        // then merge use_default_config=true so backend uses the full config as-is
        try {
          const parsed = JSON.parse($configJson.value);
          body = {
            folder_key: (parsed.data && parsed.data.folder_key) || '',
            filename: (parsed.data && parsed.data.filename) || '',
            model_type: (parsed.model && parsed.model.type) || 'xgboost',
            target_column: parsed.target_column || 'rain_total',
            use_default_config: true,
          };
        } catch (err) {
          alert('❌ Config JSON không hợp lệ: ' + err.message);
          return;
        }
      } else {
        // Quick config
        const selectedModel = getSelectedModelType();
        body = {
          folder_key: $folderKey?.value || '',
          filename: $fileName?.value || '',
          model_type: selectedModel,
          target_column: $targetColumn?.value || 'rain_total',
          test_size: parseFloat($testSize?.value || 0.15),
          valid_size: parseFloat($validSize?.value || 0.15),
          sort_by_time: document.getElementById('sortByTime')?.checked ?? true,
          feature_selection_enabled: document.getElementById('featureSelectionEnabled')?.checked ?? false,
          forecast_horizon: parseInt(document.getElementById('forecastHorizon')?.value || 24, 10),
          use_default_config: $useDefault?.checked || false,
        };
        // Stacking-specific params
        if (selectedModel === 'stacking_ensemble') {
          body.stacking_n_splits = parseInt(document.getElementById('stackingNSplits')?.value || 5, 10);
          body.stacking_predict_threshold = parseFloat(document.getElementById('stackingPredictThreshold')?.value || 0.40);
          body.stacking_rain_threshold = parseFloat(document.getElementById('stackingRainThreshold')?.value || 0.10);
          body.stacking_seed = parseInt(document.getElementById('stackingSeed')?.value || 42, 10);
        }
      }

      if (!body.folder_key && !body.data?.folder_key) {
        alert('⚠️ Vui lòng chọn thư mục dữ liệu');
        return;
      }
      if (!body.filename && !body.data?.filename) {
        alert('⚠️ Vui lòng chọn file dữ liệu');
        return;
      }

      // Disable button
      $btnStart.disabled = true;
      $btnStart.textContent = '⏳ Đang khởi chạy...';

      try {
        const resp = await fetch('/train/start/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': CSRF,
          },
          body: JSON.stringify(body),
        });

        const data = await resp.json();

        if (!data.ok) {
          alert('❌ ' + (data.error || 'Lỗi không xác định'));
          $btnStart.disabled = false;
          $btnStart.textContent = '🚀 Bắt đầu huấn luyện';
          return;
        }

        currentJobId = data.job_id;
        showLogs();
        startPolling();
      } catch (err) {
        alert('❌ Network error: ' + err.message);
        $btnStart.disabled = false;
        $btnStart.textContent = '🚀 Bắt đầu huấn luyện';
      }
    });
  }

  // ═══════════════════════════════════════════════
  // Log polling
  // ═══════════════════════════════════════════════
  function showLogs() {
    if ($sectionLogs) $sectionLogs.style.display = '';
    if ($logPre) $logPre.textContent = '';
    if ($trainStatus) {
      $trainStatus.textContent = '⏳ Đang huấn luyện...';
      $trainStatus.className = 'train-status';
    }
    $sectionLogs?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  let logAfter = 0;

  function startPolling() {
    logAfter = 0;
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollLogs, 1500);
    pollLogs(); // immediately
  }

  async function pollLogs() {
    if (!currentJobId) return;

    try {
      const resp = await fetch(`/train/tail/?job_id=${currentJobId}&after=${logAfter}`);
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
        $trainStatus.textContent = '✅ Hoàn tất!';
        $trainStatus.className = 'train-status train-status--done';
        $btnStart.disabled = false;
        $btnStart.textContent = '🚀 Bắt đầu huấn luyện';

        // Fetch & hiển thị kết quả ngay tại chỗ (không reload)
        fetchAndShowArtifacts();
      } else if (data.status === 'error') {
        clearInterval(pollTimer);
        $trainStatus.textContent = '❌ Lỗi!';
        $trainStatus.className = 'train-status train-status--error';
        $btnStart.disabled = false;
        $btnStart.textContent = '🚀 Bắt đầu huấn luyện';
      }
    } catch (err) {
      console.error('Poll error:', err);
    }
  }

  // ═══════════════════════════════════════════════
  // Fetch & hiển thị kết quả sau khi train/tune xong
  // ═══════════════════════════════════════════════
  async function fetchAndShowArtifacts() {
    // Determine which artifact endpoint to query based on lastly selected model
    const lastModel = getSelectedModelType();
    const isStacking = lastModel === 'stacking_ensemble';
    const artifactUrl = isStacking ? '/train/artifacts/?type=stacking' : '/train/artifacts/';
    try {
      const resp = await fetch(artifactUrl);
      const data = await resp.json();
      if (!data.ok || !data.artifacts || !data.artifacts.metrics) return;

      const m = data.artifacts.metrics;
      const a = data.artifacts;

      // Helper: render 1 metric card
      function metricCard(label, cssClass, metrics) {
        let rainAcc = '';
        if (metrics.Rain_Detection_Accuracy != null) {
          rainAcc = `<div class="metric-item">
            <span class="metric-name">Rain Acc</span>
            <span class="metric-value metric-value--accent">${Number(metrics.Rain_Detection_Accuracy).toFixed(4)}</span>
          </div>`;
        }
        return `<div class="metric-card ${cssClass}">
          <div class="metric-card__label">${label}</div>
          <div class="metric-card__items">
            <div class="metric-item">
              <span class="metric-name">R²</span>
              <span class="metric-value metric-value--good">${Number(metrics.R2).toFixed(4)}</span>
            </div>
            <div class="metric-item">
              <span class="metric-name">RMSE</span>
              <span class="metric-value">${Number(metrics.RMSE).toFixed(4)}</span>
            </div>
            <div class="metric-item">
              <span class="metric-name">MAE</span>
              <span class="metric-value">${Number(metrics.MAE).toFixed(4)}</span>
            </div>
            ${rainAcc}
          </div>
        </div>`;
      }

      let diagHtml = '';
      if (m.diagnostics) {
        const d = m.diagnostics;
        const statusCls = d.overfit_status === 'good' ? 'metric-badge--good'
          : d.overfit_status === 'overfit' ? 'metric-badge--warn' : 'metric-badge--bad';
        const qualCls = d.model_quality === 'excellent' ? 'metric-badge--good'
          : d.model_quality === 'good' ? 'metric-badge--good'
          : d.model_quality === 'fair' ? 'metric-badge--warn' : 'metric-badge--bad';
        diagHtml = `<div class="metric-card metric-card--diag">
          <div class="metric-card__label">DIAGNOSTICS</div>
          <div class="metric-card__items">
            <div class="metric-item">
              <span class="metric-name">Generalization</span>
              <span class="metric-value metric-badge ${statusCls}">${(d.overfit_status || '').toUpperCase()}</span>
            </div>
            <div class="metric-item">
              <span class="metric-name">Quality</span>
              <span class="metric-value metric-badge ${qualCls}">${(d.model_quality || '').toUpperCase()}</span>
            </div>
            <div class="metric-item">
              <span class="metric-name">Features</span>
              <span class="metric-value">${d.n_features_after_selection || ''}</span>
            </div>
            <div class="metric-item">
              <span class="metric-name">Zero ratio</span>
              <span class="metric-value">${d.target_zero_ratio != null ? Number(d.target_zero_ratio).toFixed(2) : ''}</span>
            </div>
          </div>
        </div>`;
      }

      const html = `
        <div class="card__header">
          <h2 class="card__title">${isStacking ? '🔥 Kết quả Stacking Ensemble' : '📊 Kết quả huấn luyện'}</h2>
        </div>
        <div class="artifacts-body">
          <div class="metrics-grid">
            ${m.train ? metricCard('TRAIN', 'metric-card--train', m.train) : ''}
            ${m.valid ? metricCard('VALIDATION', 'metric-card--valid', m.valid) : ''}
            ${m.test ? metricCard('TEST', 'metric-card--test', m.test) : ''}
            ${diagHtml}
          </div>
          <div class="artifacts-meta" style="margin-top:12px">
            <span class="artifacts-meta__item">🕒 Trained at: ${m.generated_at || ''}</span>
            <span class="artifacts-meta__item">🤖 Model: ${isStacking ? 'STACKING ENSEMBLE' : (m.model_type || '').toUpperCase()}</span>
            ${a.model_exists ? `<span class="artifacts-meta__item">💾 Size: ${a.model_size_mb} MB</span>` : ''}
          </div>
        </div>`;

      // Cập nhật hoặc tạo section artifacts
      const sectionId = isStacking ? 'sectionStackingArtifacts' : 'sectionArtifacts';
      const sectionClass = isStacking ? 'card card--artifacts stacking-artifacts-card' : 'card card--artifacts';
      let $section = document.getElementById(sectionId);
      if (!$section) {
        $section = document.createElement('section');
        $section.id = sectionId;
        $section.className = sectionClass;
        // Chèn ngay sau section logs
        const $logs = document.getElementById('sectionLogs');
        if ($logs && $logs.parentNode) {
          $logs.parentNode.insertBefore($section, $logs.nextSibling);
        } else {
          document.querySelector('.train-content')?.appendChild($section);
        }
      }
      $section.innerHTML = html;
      $section.style.display = '';

      // Scroll đến kết quả
      $section.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
      console.error('Failed to fetch artifacts:', err);
    }
  }

  // ═══════════════════════════════════════════════
  // Reset form
  // ═══════════════════════════════════════════════
  if ($btnReset) {
    $btnReset.addEventListener('click', () => {
      if ($folderKey) $folderKey.value = '';
      if ($fileName) $fileName.innerHTML = '<option value="">-- Chọn thư mục trước --</option>';
      // Reset model dropdown to ensemble
      if ($modelTypeSelect) {
        $modelTypeSelect.value = 'ensemble';
        syncModelTypeSelect('ensemble');
        updateStackingPanel();
      }
      // Reset stacking panel inputs
      const $nSplits = document.getElementById('stackingNSplits');
      const $pThresh = document.getElementById('stackingPredictThreshold');
      const $rThresh = document.getElementById('stackingRainThreshold');
      const $seed    = document.getElementById('stackingSeed');
      if ($nSplits) $nSplits.value = '5';
      if ($pThresh) $pThresh.value = '0.40';
      if ($rThresh) $rThresh.value = '0.10';
      if ($seed)    $seed.value    = '42';
      if ($targetColumn) $targetColumn.value = 'rain_total';
      if ($testSize) $testSize.value = '0.15';
      if ($validSize) $validSize.value = '0.15';
      if ($useDefault) $useDefault.checked = false;
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
        const resp = await fetch('/train/configs/');
        const data = await resp.json();

        if (data.ok && data.datasets) {
          window.__DATASETS__ = data.datasets;
          // Rebuild table
          const tbody = document.querySelector('#datasetTable tbody');
          if (tbody) {
            tbody.innerHTML = data.datasets.map(ds => `
              <tr data-folder="${ds.folder_key}" data-file="${ds.filename}">
                <td><span class="badge badge--folder">${ds.folder_key}</span></td>
                <td class="td-filename" title="${ds.filename}">${ds.filename}</td>
                <td>${ds.size_mb} MB</td>
                <td>${ds.mtime}</td>
                <td>
                  <button type="button" class="pill small btn-use-dataset"
                          data-folder="${ds.folder_key}" data-file="${ds.filename}">
                    ✅ Dùng
                  </button>
                </td>
              </tr>
            `).join('');

            // Re-attach click handlers
            tbody.querySelectorAll('.btn-use-dataset').forEach(btn => {
              btn.addEventListener('click', () => {
                $folderKey.value = btn.dataset.folder;
                updateFileDropdown(btn.dataset.folder);
                setTimeout(() => { $fileName.value = btn.dataset.file; }, 50);
                document.getElementById('sectionConfig')?.scrollIntoView({ behavior: 'smooth' });
              });
            });
          }
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

  // ═══════════════════════════════════════════════
  // Optuna Tune logic
  // ═══════════════════════════════════════════════
  const $btnStartTune = document.getElementById('btnStartTune');
  const $tuneProgressWrap = document.getElementById('tuneProgressWrap');
  const $tuneProgressFill = document.getElementById('tuneProgressFill');
  const $tuneProgressPercent = document.getElementById('tuneProgressPercent');
  const $tuneProgressStep = document.getElementById('tuneProgressStep');
  const $tuneLogPre = document.getElementById('tuneLogPre');
  let tuneJobId = null;
  let tunePollTimer = null;
  let tuneLogAfter = 0;

  if ($btnStartTune) {
    $btnStartTune.addEventListener('click', async () => {
      const trials = parseInt(document.getElementById('tuneTrials')?.value || 100, 10);
      const metric = document.getElementById('tuneMetric')?.value || 'rain_acc';
      const autoApply = document.getElementById('tuneAutoApply')?.checked ?? true;

      $btnStartTune.disabled = true;
      $btnStartTune.textContent = '⏳ Đang khởi chạy...';

      if ($tuneProgressWrap) $tuneProgressWrap.style.display = '';
      if ($tuneLogPre) $tuneLogPre.textContent = '';
      if ($tuneProgressFill) $tuneProgressFill.style.width = '0%';
      if ($tuneProgressPercent) $tuneProgressPercent.textContent = '0%';
      if ($tuneProgressStep) $tuneProgressStep.textContent = 'Khởi tạo';

      try {
        const resp = await fetch('/train/tune/start/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF },
          body: JSON.stringify({ trials, metric, auto_apply: autoApply }),
        });
        const data = await resp.json();

        if (!data.ok) {
          alert('❌ ' + (data.error || 'Lỗi không xác định'));
          $btnStartTune.disabled = false;
          $btnStartTune.textContent = '🔬 Bắt đầu tối ưu';
          return;
        }

        tuneJobId = data.job_id;
        tuneLogAfter = 0;
        if (tunePollTimer) clearInterval(tunePollTimer);
        tunePollTimer = setInterval(pollTuneLogs, 1500);
        pollTuneLogs();
      } catch (err) {
        alert('❌ Network error: ' + err.message);
        $btnStartTune.disabled = false;
        $btnStartTune.textContent = '🔬 Bắt đầu tối ưu';
      }
    });
  }

  async function pollTuneLogs() {
    if (!tuneJobId) return;
    try {
      const resp = await fetch(`/train/tune/tail/?job_id=${tuneJobId}&after=${tuneLogAfter}`);
      const data = await resp.json();
      if (!data.ok) return;

      if (data.logs && data.logs.length > 0) {
        data.logs.forEach(line => { if ($tuneLogPre) $tuneLogPre.textContent += line + '\n'; });
        tuneLogAfter = data.total;
        const logOutput = $tuneLogPre?.parentElement;
        if (logOutput) logOutput.scrollTop = logOutput.scrollHeight;
      }

      if ($tuneProgressFill) $tuneProgressFill.style.width = data.progress + '%';
      if ($tuneProgressPercent) $tuneProgressPercent.textContent = data.progress + '%';
      if ($tuneProgressStep) $tuneProgressStep.textContent = data.step || '';

      if (data.status === 'done') {
        clearInterval(tunePollTimer);
        if ($btnStartTune) {
          $btnStartTune.disabled = false;
          $btnStartTune.textContent = '🔬 Bắt đầu tối ưu';
        }
        // Fetch & hiển thị kết quả ngay tại chỗ (không reload)
        fetchAndShowArtifacts();
      } else if (data.status === 'error') {
        clearInterval(tunePollTimer);
        if ($btnStartTune) {
          $btnStartTune.disabled = false;
          $btnStartTune.textContent = '🔬 Bắt đầu tối ưu';
        }
      }
    } catch (err) {
      console.error('Tune poll error:', err);
    }
  }

  // Toggle best params card
  document.getElementById('btnToggleBestParams')?.addEventListener('click', function () {
    const body = document.getElementById('bestParamsBody');
    if (body) {
      const hidden = body.style.display === 'none';
      body.style.display = hidden ? '' : 'none';
      this.textContent = hidden ? 'Thu gọn' : 'Mở rộng';
    }
  });

  // Toggle tune section
  document.getElementById('btnToggleTune')?.addEventListener('click', function () {
    const body = document.getElementById('tuneBody');
    if (body) {
      const hidden = body.style.display === 'none';
      body.style.display = hidden ? '' : 'none';
      this.textContent = hidden ? 'Thu gọn' : 'Mở rộng';
    }
  });
});
