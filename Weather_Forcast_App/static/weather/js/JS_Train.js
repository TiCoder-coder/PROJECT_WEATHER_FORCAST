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
  const $folderKey     = document.getElementById('folderKey');
  const $fileName      = document.getElementById('fileName');
  const $modelType     = document.getElementById('modelType');
  const $targetColumn  = document.getElementById('targetColumn');
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
        body = {
          folder_key: $folderKey?.value || '',
          filename: $fileName?.value || '',
          model_type: $modelType?.value || 'xgboost',
          target_column: $targetColumn?.value || 'rain_total',
          test_size: parseFloat($testSize?.value || 0.15),
          valid_size: parseFloat($validSize?.value || 0.15),
          use_default_config: $useDefault?.checked || false,
        };
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

        // Reload page after 2s to refresh artifacts
        setTimeout(() => location.reload(), 2500);
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
  // Reset form
  // ═══════════════════════════════════════════════
  if ($btnReset) {
    $btnReset.addEventListener('click', () => {
      if ($folderKey) $folderKey.value = '';
      if ($fileName) $fileName.innerHTML = '<option value="">-- Chọn thư mục trước --</option>';
      if ($modelType) $modelType.value = 'ensemble';
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
});
