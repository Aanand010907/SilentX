/* =====================================================================
   SILENTX // TRUSTGUARD DASHBOARD APPLICATION LOGIC
   Vanilla JS Engine for Streaming Monitoring, Replay, and Auditing
   ===================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Ensure data exists
  if (!window.SILENTX_DATA) {
    console.error('SILENTX_DATA not found. Please ensure data.js is loaded.');
    return;
  }

  const DATA = window.SILENTX_DATA;
  const elecWindows = DATA.electricity_windows || [];
  const synthSummary = DATA.synthetic_summary || [];
  const statTests = DATA.statistical_tests || [];
  const impossibility = DATA.concept_shift_impossibility || [];

  // =====================================================================
  // STATE MANAGEMENT
  // =====================================================================
  let currentWindowIdx = 0;
  let isPlaying = false;
  let playInterval = null;
  const playSpeedMs = 1000;
  let currentFilter = 'all';

  // =====================================================================
  // TAB NAVIGATION
  // =====================================================================
  const tabs = document.querySelectorAll('.nav-tab');
  const tabContents = document.querySelectorAll('.tab-content');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTab = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove('active'));
      tabContents.forEach(tc => tc.classList.remove('active'));

      tab.classList.add('active');
      const targetEl = document.getElementById(`tab-${targetTab}`);
      if (targetEl) targetEl.classList.add('active');

      // Trigger chart re-render if switching to electricity tab
      if (targetTab === 'electricity') {
        renderStreamChart();
      }
    });
  });

  // =====================================================================
  // ELECTRICITY STREAM CONTROLS
  // =====================================================================
  const slider = document.getElementById('window-slider');
  const sliderVal = document.getElementById('slider-val');
  const btnPlay = document.getElementById('btn-play');
  const playIcon = document.getElementById('play-icon');
  const playText = document.getElementById('play-text');
  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');
  const btnReset = document.getElementById('btn-reset');
  const filterSelect = document.getElementById('window-filter');

  function updateWindowDisplay(idx) {
    if (idx < 0) idx = 0;
    if (idx >= elecWindows.length) idx = elecWindows.length - 1;
    currentWindowIdx = idx;

    slider.value = idx;
    sliderVal.textContent = `Window ${idx} / ${elecWindows.length - 1}`;

    const win = elecWindows[idx];
    if (!win) return;

    // Hero Header
    const heroTitle = document.getElementById('hero-title');
    heroTitle.innerHTML = `Window #${win.window_id} <span class="text-muted">(${win.start_sample.toLocaleString()} – ${win.end_sample.toLocaleString()})</span>`;

    // Alert Badge
    const alertBadge = document.getElementById('hero-alert-badge');
    const isDegraded = win.is_degraded || win.true_degradation >= 0.05;
    const isAlarm = win.tg_alarm;

    if (isAlarm) {
      alertBadge.className = 'alert-badge badge-alarm';
      alertBadge.textContent = '🚨 TRUSTGUARD CRITICAL ALARM';
    } else if (win.true_degradation < 0.02) {
      alertBadge.className = 'alert-badge badge-normal';
      alertBadge.textContent = 'STATIONARY / BENIGN WINDOW';
    } else {
      alertBadge.className = 'alert-badge badge-amber';
      alertBadge.textContent = 'ELEVATED MONITORING AMBIGUITY';
    }

    // Hero Metrics
    const valTrueAcc = document.getElementById('val-true-acc');
    const valDeg = document.getElementById('val-deg');
    const valTgRisk = document.getElementById('val-tg-risk');
    const valAlarmStatus = document.getElementById('val-alarm-status');
    const valCottEst = document.getElementById('val-cott-est');
    const valCottErr = document.getElementById('val-cott-err');
    const valCbpeEst = document.getElementById('val-cbpe-est');
    const valCbpeErr = document.getElementById('val-cbpe-err');

    const trueAccPct = (win.true_accuracy * 100).toFixed(1);
    valTrueAcc.textContent = `${trueAccPct}%`;
    valTrueAcc.className = isDegraded ? 'metric-num text-rose' : 'metric-num text-emerald';

    const degPct = (win.true_degradation * 100).toFixed(1);
    valDeg.textContent = `Δ ${degPct}%`;
    valDeg.className = isDegraded ? 'metric-diff text-rose' : 'metric-diff text-muted';

    valTgRisk.textContent = win.tg_risk.toFixed(3);
    valTgRisk.className = isAlarm ? 'metric-num text-rose' : 'metric-num text-cyan';

    valAlarmStatus.textContent = isAlarm ? '🚨 Triggered Degradation Alarm' : '✓ Below Alarm Threshold (τ = 0.041)';
    valAlarmStatus.className = isAlarm ? 'metric-sub text-rose font-bold' : 'metric-sub text-emerald';

    // COTT vs CBPE
    const cottAccPct = (win.est_cott * 100).toFixed(1);
    const cottErr = Math.abs(win.est_cott - win.true_accuracy) * 100;
    valCottEst.textContent = `${cottAccPct}%`;
    valCottErr.textContent = `Err: ${cottErr.toFixed(1)}%`;
    valCottErr.className = cottErr < 5.0 ? 'metric-diff text-emerald' : 'metric-diff text-amber';

    const cbpeAccPct = (win.est_cbpe * 100).toFixed(1);
    const cbpeErr = (win.est_cbpe - win.true_accuracy) * 100;
    valCbpeEst.textContent = `${cbpeAccPct}%`;
    valCbpeErr.textContent = `Err: ${cbpeErr > 0 ? '+' : ''}${cbpeErr.toFixed(1)}%`;
    valCbpeErr.className = Math.abs(cbpeErr) < 5.0 ? 'metric-diff text-emerald' : 'metric-diff text-rose font-bold';

    // Multi-Signal Progress Bars
    updateProgressBar('bar-fill-attr', 'bar-val-attr', win.s_attr, 1.0);
    updateProgressBar('bar-fill-dis', 'bar-val-dis', win.s_dis, 0.6);
    updateProgressBar('bar-fill-conf', 'bar-val-conf', win.s_conf, 0.6);
    updateProgressBar('bar-fill-grad', 'bar-val-grad', win.s_grad, 0.1);
    updateProgressBar('bar-fill-d3m', 'bar-val-d3m', win.s_d3m, 0.8);

    // Paradox Inspector
    document.getElementById('val-ks-max').textContent = win.ks_max.toFixed(4);

    // Highlight row in table
    highlightTableRow(idx);
  }

  function updateProgressBar(barId, valId, value, maxScale) {
    const barEl = document.getElementById(barId);
    const valEl = document.getElementById(valId);
    if (!barEl || !valEl) return;

    valEl.textContent = value.toFixed(3);
    const pct = Math.min(100, Math.max(0, (value / maxScale) * 100));
    barEl.style.width = `${pct}%`;
  }

  // Play / Pause Controls
  btnPlay.addEventListener('click', () => {
    if (isPlaying) {
      pauseReplay();
    } else {
      startReplay();
    }
  });

  function startReplay() {
    isPlaying = true;
    playIcon.textContent = '⏸';
    playText.textContent = 'Pause Replay';
    btnPlay.classList.add('pulse-emerald');

    playInterval = setInterval(() => {
      let nextIdx = currentWindowIdx + 1;
      if (nextIdx >= elecWindows.length) {
        nextIdx = 0;
      }
      updateWindowDisplay(nextIdx);
    }, playSpeedMs);
  }

  function pauseReplay() {
    isPlaying = false;
    playIcon.textContent = '▶';
    playText.textContent = 'Auto Replay';
    btnPlay.classList.remove('pulse-emerald');
    if (playInterval) clearInterval(playInterval);
  }

  btnPrev.addEventListener('click', () => {
    pauseReplay();
    updateWindowDisplay(currentWindowIdx - 1);
  });

  btnNext.addEventListener('click', () => {
    pauseReplay();
    updateWindowDisplay(currentWindowIdx + 1);
  });

  btnReset.addEventListener('click', () => {
    pauseReplay();
    updateWindowDisplay(0);
  });

  slider.addEventListener('input', (e) => {
    pauseReplay();
    updateWindowDisplay(parseInt(e.target.value, 10));
  });

  filterSelect.addEventListener('change', (e) => {
    currentFilter = e.target.value;
    filterElectricityTable();
  });

  // =====================================================================
  // SVG STREAM CHART RENDERING
  // =====================================================================
  function renderStreamChart() {
    const svg = document.getElementById('stream-chart');
    if (!svg || elecWindows.length === 0) return;

    const width = 1000;
    const height = 320;
    const padX = 50;
    const padY = 30;
    const plotW = width - 2 * padX;
    const plotH = height - 2 * padY;

    const n = elecWindows.length;
    const xStep = plotW / (n - 1);

    // Map functions
    const getX = (i) => padX + i * xStep;
    const getY = (val) => padY + (1.0 - Math.min(1.0, Math.max(0.0, val))) * plotH;

    // Build paths
    let trueAccPath = '';
    let cottPath = '';
    let scalarOtPath = '';
    let atcPath = '';
    let cbpePath = '';
    let riskAreaPath = `M ${getX(0)} ${getY(0)}`;

    elecWindows.forEach((w, i) => {
      const x = getX(i);
      const yTrue = getY(w.true_accuracy);
      const yCott = getY(w.est_cott);
      const yScalar = getY(w.est_scalar_ot);
      const yAtc = getY(w.est_atc);
      const yCbpe = getY(w.est_cbpe);
      const yRisk = getY(w.tg_risk * 1.5); // scaled for visual prominence

      if (i === 0) {
        trueAccPath += `M ${x} ${yTrue}`;
        cottPath += `M ${x} ${yCott}`;
        scalarOtPath += `M ${x} ${yScalar}`;
        atcPath += `M ${x} ${yAtc}`;
        cbpePath += `M ${x} ${yCbpe}`;
        riskAreaPath += ` L ${x} ${yRisk}`;
      } else {
        trueAccPath += ` L ${x} ${yTrue}`;
        cottPath += ` L ${x} ${yCott}`;
        scalarOtPath += ` L ${x} ${yScalar}`;
        atcPath += ` L ${x} ${yAtc}`;
        cbpePath += ` L ${x} ${yCbpe}`;
        riskAreaPath += ` L ${x} ${yRisk}`;
      }
    });

    riskAreaPath += ` L ${getX(n - 1)} ${getY(0)} Z`;

    // Grid lines
    let gridLinesSvg = '';
    [0.2, 0.4, 0.6, 0.8, 1.0].forEach(level => {
      const y = getY(level);
      gridLinesSvg += `
        <line x1="${padX}" y1="${y}" x2="${width - padX}" y2="${y}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="4" />
        <text x="${padX - 8}" y="${y + 4}" fill="#64748b" font-size="10" font-family="JetBrains Mono" text-anchor="end">${Math.round(level * 100)}%</text>
      `;
    });

    // Window ticks
    for (let i = 0; i < n; i += 5) {
      const x = getX(i);
      gridLinesSvg += `
        <line x1="${x}" y1="${padY}" x2="${x}" y2="${height - padY}" stroke="rgba(255,255,255,0.04)" />
        <text x="${x}" y="${height - padY + 16}" fill="#64748b" font-size="10" font-family="JetBrains Mono" text-anchor="middle">W${i}</text>
      `;
    }

    svg.innerHTML = `
      ${gridLinesSvg}
      <!-- Risk Area Shading -->
      <path d="${riskAreaPath}" fill="rgba(56, 189, 248, 0.12)" />
      
      <!-- Estimator Curves -->
      <path d="${cbpePath}" fill="none" stroke="#ef4444" stroke-width="2" opacity="0.75" />
      <path d="${atcPath}" fill="none" stroke="#a855f7" stroke-width="2" opacity="0.8" />
      <path d="${scalarOtPath}" fill="none" stroke="#38bdf8" stroke-width="2" opacity="0.85" />
      <path d="${cottPath}" fill="none" stroke="#10b981" stroke-width="2.5" />
      
      <!-- Ground Truth Curve (Gold dashed) -->
      <path d="${trueAccPath}" fill="none" stroke="#fbbf24" stroke-width="2.5" stroke-dasharray="6,4" />

      <!-- Interactive Crosshair (Hidden by default) -->
      <line id="crosshair-x" x1="0" y1="${padY}" x2="0" y2="${height - padY}" stroke="rgba(56, 189, 248, 0.8)" stroke-width="1.5" stroke-dasharray="2,2" style="display:none;" />
      <circle id="crosshair-point" cx="0" cy="0" r="5" fill="#38bdf8" stroke="#fff" stroke-width="2" style="display:none;" />
    `;

    // Tooltip and click interaction
    const container = document.getElementById('chart-container');
    const tooltip = document.getElementById('chart-tooltip');
    const crossX = document.getElementById('crosshair-x');
    const crossPt = document.getElementById('crosshair-point');

    svg.addEventListener('mousemove', (e) => {
      const rect = svg.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const relX = (mouseX / rect.width) * width;

      if (relX >= padX && relX <= width - padX) {
        const approxIdx = Math.round((relX - padX) / xStep);
        if (approxIdx >= 0 && approxIdx < n) {
          const w = elecWindows[approxIdx];
          const xPos = getX(approxIdx);
          const yPos = getY(w.true_accuracy);

          crossX.setAttribute('x1', xPos);
          crossX.setAttribute('x2', xPos);
          crossX.style.display = 'block';

          crossPt.setAttribute('cx', xPos);
          crossPt.setAttribute('cy', yPos);
          crossPt.style.display = 'block';

          tooltip.style.display = 'block';
          tooltip.style.left = `${(xPos / width) * rect.width + 12}px`;
          tooltip.style.top = `${(yPos / height) * rect.height - 40}px`;
          tooltip.innerHTML = `
            <strong>Window #${w.window_id}</strong><br/>
            True Acc: <span style="color:#fbbf24">${(w.true_accuracy * 100).toFixed(1)}%</span><br/>
            COTT: <span style="color:#10b981">${(w.est_cott * 100).toFixed(1)}%</span> |
            Scalar OT: <span style="color:#38bdf8">${(w.est_scalar_ot * 100).toFixed(1)}%</span><br/>
            ATC: <span style="color:#a855f7">${(w.est_atc * 100).toFixed(1)}%</span> |
            CBPE: <span style="color:#ef4444">${(w.est_cbpe * 100).toFixed(1)}%</span><br/>
            TG Risk: <span style="color:#38bdf8">${w.tg_risk.toFixed(3)}</span> (${w.tg_alarm ? '🚨 ALARM' : 'OK'})
          `;
        }
      }
    });

    svg.addEventListener('mouseleave', () => {
      crossX.style.display = 'none';
      crossPt.style.display = 'none';
      tooltip.style.display = 'none';
    });

    svg.addEventListener('click', (e) => {
      const rect = svg.getBoundingClientRect();
      const relX = ((e.clientX - rect.left) / rect.width) * width;
      if (relX >= padX && relX <= width - padX) {
        const clickedIdx = Math.round((relX - padX) / xStep);
        if (clickedIdx >= 0 && clickedIdx < n) {
          pauseReplay();
          updateWindowDisplay(clickedIdx);
        }
      }
    });
  }

  // =====================================================================
  // TABLE POPULATION
  // =====================================================================
  function populateElectricityTable() {
    const tbody = document.getElementById('tbody-electricity');
    if (!tbody) return;
    tbody.innerHTML = '';

    elecWindows.forEach((w) => {
      const tr = document.createElement('tr');
      tr.id = `row-elec-${w.window_id}`;
      tr.dataset.degraded = w.is_degraded ? 'true' : 'false';
      tr.dataset.alarm = w.tg_alarm ? 'true' : 'false';
      tr.dataset.benign = w.true_degradation < 0.02 ? 'true' : 'false';

      tr.innerHTML = `
        <td class="font-bold text-cyan">W${w.window_id}</td>
        <td>${w.start_sample} – ${w.end_sample}</td>
        <td class="${w.is_degraded ? 'text-rose' : 'text-emerald'}">${(w.true_accuracy * 100).toFixed(1)}%</td>
        <td class="${w.is_degraded ? 'text-rose font-bold' : 'text-muted'}">${(w.true_degradation * 100).toFixed(1)}%</td>
        <td>${w.is_degraded ? '<span class="badge badge-rose">DEGRADED</span>' : '<span class="badge badge-subtle">NORMAL</span>'}</td>
        <td class="${w.tg_alarm ? 'text-rose font-bold' : 'text-cyan'}">${w.tg_risk.toFixed(3)}</td>
        <td>${w.tg_alarm ? '<span class="badge badge-alarm">ALARM</span>' : '<span class="text-emerald">✓</span>'}</td>
        <td class="text-emerald">${(w.est_cott * 100).toFixed(1)}%</td>
        <td class="text-sky">${(w.est_scalar_ot * 100).toFixed(1)}%</td>
        <td class="text-purple">${(w.est_atc * 100).toFixed(1)}%</td>
        <td class="text-rose">${(w.est_cbpe * 100).toFixed(1)}%</td>
        <td class="text-muted">${w.ks_max.toFixed(3)}</td>
      `;

      tr.addEventListener('click', () => {
        pauseReplay();
        updateWindowDisplay(w.window_id);
      });

      tbody.appendChild(tr);
    });
  }

  function highlightTableRow(idx) {
    const rows = document.querySelectorAll('#tbody-electricity tr');
    rows.forEach(r => r.classList.remove('row-selected'));
    const targetRow = document.getElementById(`row-elec-${idx}`);
    if (targetRow) {
      targetRow.classList.add('row-selected');
      // Gentle scroll if off-screen
      targetRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function filterElectricityTable() {
    const rows = document.querySelectorAll('#tbody-electricity tr');
    rows.forEach(row => {
      let show = true;
      if (currentFilter === 'degraded' && row.dataset.degraded !== 'true') show = false;
      if (currentFilter === 'benign' && row.dataset.benign !== 'true') show = false;
      if (currentFilter === 'alarm' && row.dataset.alarm !== 'true') show = false;
      row.style.display = show ? '' : 'none';
    });
  }

  // Populate Synthetic Table
  function populateSyntheticTable() {
    const tbody = document.getElementById('tbody-synthetic');
    if (!tbody) return;
    tbody.innerHTML = '';

    synthSummary.forEach(row => {
      const tr = document.createElement('tr');
      tr.dataset.seed = row.seed;

      tr.innerHTML = `
        <td class="mono font-bold">${row.seed}</td>
        <td class="mono">W${row.window_id}</td>
        <td><strong>${row.regime}</strong></td>
        <td class="mono">${row.severity.toFixed(1)}</td>
        <td class="${row.is_degraded ? 'text-rose' : 'text-emerald'}">${(row.true_accuracy * 100).toFixed(1)}%</td>
        <td class="${row.is_degraded ? 'text-rose' : 'text-muted'}">${(row.true_degradation * 100).toFixed(1)}%</td>
        <td class="${row.tg_alarm ? 'text-rose font-bold' : 'text-cyan'}">${row.tg_risk.toFixed(3)}</td>
        <td>${row.tg_alarm ? '<span class="badge badge-alarm">ALARM</span>' : '<span class="text-emerald">✓</span>'}</td>
        <td class="text-purple">${(row.est_atc * 100).toFixed(1)}%</td>
        <td class="text-emerald">${(row.est_cott * 100).toFixed(1)}%</td>
        <td class="text-rose">${(row.est_cbpe * 100).toFixed(1)}%</td>
        <td class="mono">${row.s_attr.toFixed(3)}</td>
        <td class="mono">${row.s_dis.toFixed(3)}</td>
      `;
      tbody.appendChild(tr);
    });

    const seedFilter = document.getElementById('synth-seed-filter');
    seedFilter.addEventListener('change', (e) => {
      const selectedSeed = e.target.value;
      const rows = tbody.querySelectorAll('tr');
      rows.forEach(r => {
        r.style.display = (selectedSeed === 'all' || r.dataset.seed === selectedSeed) ? '' : 'none';
      });
    });
  }

  // Populate Impossibility Table
  function populateImpossibilityTable() {
    const tbody = document.getElementById('tbody-impossibility');
    if (!tbody) return;
    tbody.innerHTML = '';

    impossibility.forEach(w => {
      const tr = document.createElement('tr');
      const isCollapse = w.world.includes('World_B');

      tr.innerHTML = `
        <td class="${isCollapse ? 'text-rose font-bold' : 'text-emerald font-bold'}">${w.world}</td>
        <td class="${isCollapse ? 'text-rose font-bold' : 'text-emerald font-bold'}">${(w.true_accuracy * 100).toFixed(1)}%</td>
        <td class="${isCollapse ? 'text-rose font-bold' : 'text-muted'}">${(w.true_degradation * 100).toFixed(1)}%</td>
        <td class="text-cyan mono">${w.tg_risk.toFixed(4)}</td>
        <td>${w.tg_alarm ? '<span class="badge badge-alarm">ALARM</span>' : '<span class="badge badge-emerald">FALSE</span>'}</td>
        <td class="mono">${(w.est_cott * 100).toFixed(1)}%</td>
        <td class="mono">${(w.est_scalar_ot * 100).toFixed(1)}%</td>
        <td class="mono">${(w.est_atc * 100).toFixed(1)}%</td>
        <td class="mono">${(w.est_cbpe * 100).toFixed(1)}%</td>
        <td class="mono">${(w.est_doc * 100).toFixed(1)}%</td>
        <td class="mono">${w.ks_max.toFixed(4)}</td>
        <td class="mono">${w.psi_max.toFixed(4)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Populate Statistics Table
  function populateStatisticsTable() {
    const tbody = document.getElementById('tbody-statistics');
    if (!tbody) return;
    tbody.innerHTML = '';

    statTests.forEach(test => {
      const tr = document.createElement('tr');
      const isSignificant = test.mbb_significant_05;

      tr.innerHTML = `
        <td class="font-bold text-cyan">${test.test_pair}</td>
        <td class="text-muted">${test.statistical_unit}</td>
        <td class="mono">${test.n_windows}</td>
        <td class="mono ${test.mean_error_diff > 0 ? 'text-emerald' : 'text-muted'}">${test.mean_error_diff > 0 ? '+' : ''}${test.mean_error_diff.toFixed(4)}</td>
        <td class="mono text-muted">[${test.mbb_b3_ci_lower.toFixed(4)}, ${test.mbb_b3_ci_upper.toFixed(4)}]</td>
        <td class="mono">${test.mbb_b3_raw_p.toFixed(4)}</td>
        <td class="mono text-muted">${test.mbb_sensitivity_p_b2.toFixed(4)}</td>
        <td class="mono text-muted">${test.mbb_sensitivity_p_b5.toFixed(4)}</td>
        <td class="mono text-muted">${test.mbb_sensitivity_p_b8.toFixed(4)}</td>
        <td class="mono">${test.wilcoxon_raw_p < 0.001 ? test.wilcoxon_raw_p.toExponential(2) : test.wilcoxon_raw_p.toFixed(4)}</td>
        <td class="mono font-bold ${isSignificant ? 'text-emerald' : 'text-amber'}">${test.mbb_fdr_adj_p.toFixed(4)}</td>
        <td>${isSignificant ? '<span class="badge badge-emerald">SIGNIFICANT</span>' : '<span class="badge badge-amber">INDISTINGUISHABLE</span>'}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // =====================================================================
  // LIVE DRIFT STRESS SIMULATOR LOGIC
  // =====================================================================
  const simSliderNoise = document.getElementById('sim-slider-noise');
  const simSliderCausal = document.getElementById('sim-slider-causal');
  const simSliderCorr = document.getElementById('sim-slider-corr');
  const simValNoise = document.getElementById('sim-val-noise');
  const simValCausal = document.getElementById('sim-val-causal');
  const simValCorr = document.getElementById('sim-val-corr');

  const simOutRisk = document.getElementById('sim-out-risk');
  const simOutBadge = document.getElementById('sim-out-badge');
  const simOutAttr = document.getElementById('sim-out-attr');
  const simOutDis = document.getElementById('sim-out-dis');
  const simOutAccDrop = document.getElementById('sim-out-acc-drop');
  const simOutKs = document.getElementById('sim-out-ks');
  const simDiagnosisBox = document.getElementById('sim-diagnosis-box');

  const btnSimReset = document.getElementById('btn-sim-reset');
  const btnPresetHarmless = document.getElementById('btn-sim-preset-harmless');
  const btnPresetCollapse = document.getElementById('btn-sim-preset-collapse');

  function calculateSimulation() {
    const shiftNoise = parseFloat(simSliderNoise.value);
    const shiftCausal = parseFloat(simSliderCausal.value);
    const shiftCorr = parseFloat(simSliderCorr.value);

    simValNoise.textContent = `${shiftNoise.toFixed(1)} σ`;
    simValCausal.textContent = `${shiftCausal.toFixed(1)} σ`;
    simValCorr.textContent = shiftCorr.toFixed(1);

    // Weights from synthetic pipeline
    const wCausal = 0.86; // X0..X2
    const wNoise = 0.10;  // X5..X9
    const baseRisk = 0.035;

    // Linear KS calculations
    const ksCausal = Math.min(1.0, shiftCausal / 3.0);
    const ksNoise = Math.min(1.0, shiftNoise / 3.0);
    const ksMax = Math.max(ksCausal, ksNoise);

    // Linear attribution marginal drift
    const dMarg = wCausal * ksCausal + wNoise * ksNoise;
    const dCorr = 0.5 * shiftCorr;
    const sAttr = Math.min(1.0, dMarg + dCorr + 0.12);

    // Cross architecture disagreement (Tree saturates, MLP extrapolates on high noise)
    const sDis = Math.min(1.0, 0.10 + 0.20 * (shiftNoise / 6.0) + 0.35 * (shiftCausal / 5.0));

    // True degradation
    const trueDegradation = Math.min(0.80, 0.18 * shiftCausal + 0.08 * shiftCorr + 0.003 * shiftNoise);

    // Composite TrustGuard Risk
    const excessAttr = Math.max(0.0, sAttr - 0.12);
    const excessDis = Math.max(0.0, sDis - 0.10);
    const riskTg = Math.min(1.0, Math.max(0.0, baseRisk + 0.30 * excessAttr + 0.25 * excessDis + 0.15 * (shiftCausal / 5.0) * 0.4));

    // Threshold comparison
    const threshold = 0.041;
    const isAlarm = riskTg >= threshold;

    // Update UI
    simOutRisk.textContent = riskTg.toFixed(3);
    simOutRisk.className = isAlarm ? 'sim-risk-num text-rose' : 'sim-risk-num text-cyan';

    simOutAttr.textContent = sAttr.toFixed(3);
    simOutDis.textContent = sDis.toFixed(3);
    simOutKs.textContent = ksMax.toFixed(3);

    const degPct = (trueDegradation * 100).toFixed(1);
    simOutAccDrop.textContent = `-${degPct}%`;
    simOutAccDrop.className = trueDegradation >= 0.05 ? 'mono text-rose font-bold' : 'mono text-emerald';

    if (isAlarm) {
      simOutBadge.className = 'alert-badge badge-alarm';
      simOutBadge.textContent = '🚨 TRUSTGUARD ALARM ACTIVE';
    } else {
      simOutBadge.className = 'alert-badge badge-normal';
      simOutBadge.textContent = 'NORMAL OPERATION';
    }

    // Diagnostic interpretation
    if (shiftNoise > 3.0 && shiftCausal === 0.0 && shiftCorr === 0.0) {
      simDiagnosisBox.innerHTML = `
        <strong>Audit Diagnosis (Harmless Drift Contradiction):</strong> Even though true accuracy degradation is negligible (<span class="mono text-emerald">-${degPct}%</span>), the 4.5σ noise shift triggers unconstrained MLP probe extrapolation (S_dis = <span class="mono">${sDis.toFixed(3)}</span>), elevating composite risk past the stationary threshold (τ = 0.041). This illustrates the primary Stage 2 audit finding.
      `;
    } else if (shiftCausal > 1.5 || shiftCorr > 0.5) {
      simDiagnosisBox.innerHTML = `
        <strong>Audit Diagnosis (True Predictive Deterioration):</strong> Causal drivers displaced off the training manifold. Actual performance collapses (<span class="mono text-rose">-${degPct}%</span>). TrustGuard fires a critical alert (Risk = <span class="mono text-rose">${riskTg.toFixed(3)}</span>), successfully flagging silent model failure.
      `;
    } else {
      simDiagnosisBox.innerHTML = `
        <strong>Audit Diagnosis:</strong> The model is operating in a stationary in-distribution regime. All monitoring signals remain within reference tolerance bounds.
      `;
    }
  }

  [simSliderNoise, simSliderCausal, simSliderCorr].forEach(s => {
    s.addEventListener('input', calculateSimulation);
  });

  btnSimReset.addEventListener('click', () => {
    simSliderNoise.value = 0;
    simSliderCausal.value = 0;
    simSliderCorr.value = 0;
    calculateSimulation();
  });

  btnPresetHarmless.addEventListener('click', () => {
    simSliderNoise.value = 5.0;
    simSliderCausal.value = 0.0;
    simSliderCorr.value = 0.0;
    calculateSimulation();
  });

  btnPresetCollapse.addEventListener('click', () => {
    simSliderNoise.value = 0.0;
    simSliderCausal.value = 4.0;
    simSliderCorr.value = 0.8;
    calculateSimulation();
  });

  // =====================================================================
  // INITIALIZATION
  // =====================================================================
  populateElectricityTable();
  populateSyntheticTable();
  populateImpossibilityTable();
  populateStatisticsTable();
  updateWindowDisplay(0);
  renderStreamChart();
  calculateSimulation();

  console.log('SilentX // TrustGuard Dashboard initialized successfully.');
});
