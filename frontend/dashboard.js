/**
 * dashboard.js
 * ─────────────────────────────────────────────────────
 * Main dashboard controller.
 * Orchestrates data fetching, state management, and
 * wires all sub-modules (heatmap, shap_panel, simulator).
 * ─────────────────────────────────────────────────────
 */

'use strict';

/* ─── App-wide state ─── */
const AppState = {
  repo: '',
  developer: '',           // '' = all developers
  since: '',
  until: '',
  flowScores: [],          // raw API response array
  developers: [],          // unique developer emails
  status: 'loading',       // 'loading' | 'ready' | 'error'
  trendChart: null,        // Chart.js instance (trend line)
};

/* ─── API base URL (same origin in production) ─── */
const API_BASE = 'http://localhost:5000/api';

/* ─── Utility: fetch with JSON parse ─── */
async function apiFetch(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API error ${res.status} on ${path}`);
  return res.json();
}

/* ─── DOM helpers ─── */
const el = id => document.getElementById(id);

function setText(id, text) {
  const node = el(id);
  if (node) node.textContent = text;
}

function setHTML(id, html) {
  const node = el(id);
  if (node) node.innerHTML = html;
}

/* ═══════════════════════════════════════════════════
   INITIALISATION
═══════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', async () => {
  initDefaultDates();
  bindControls();
  await checkStatus();
  await loadDashboard();
});

/* ─── Set default date range (last 90 days) ─── */
function initDefaultDates() {
  const today = new Date();
  const ninetyDaysAgo = new Date();
  ninetyDaysAgo.setDate(today.getDate() - 90);

  const fmt = d => d.toISOString().slice(0, 10);

  el('date-until').value = fmt(today);
  el('date-since').value = fmt(ninetyDaysAgo);

  AppState.since = fmt(ninetyDaysAgo);
  AppState.until = fmt(today);
}

/* ─── Bind filter controls ─── */
function bindControls() {
  el('btn-analyse').addEventListener('click', () => loadDashboard());

  el('dev-select').addEventListener('change', e => {
    AppState.developer = e.target.value;
    rerenderHeatmap();
    renderDevBreakdown();
  });

  el('date-since').addEventListener('change', e => { AppState.since = e.target.value; });
  el('date-until').addEventListener('change', e => { AppState.until = e.target.value; });

  el('btn-simulator').addEventListener('click', () => openSimulator());
}

/* ═══════════════════════════════════════════════════
   STATUS CHECK
═══════════════════════════════════════════════════ */
async function checkStatus() {
  try {
    const res = await apiFetch('/status');
    if (res.ok) {
      updateStatusBadge('ready', '● Ready');
      AppState.status = 'ready';

      // Populate repo selector
      const repoSelect = el('repo-select');
      repoSelect.innerHTML = `<option value="${res.data.repo}">${res.data.repo}</option>`;
      AppState.repo = res.data.repo;

      // Update subtitle
      const d = res.data;
      setText('header-subtitle',
        `${d.repo} · ${d.total_developer_days.toLocaleString()} developer-days · Model: ${d.model}`);
    }
  } catch {
    updateStatusBadge('error', '● Error');
    AppState.status = 'error';
    showFallback();
  }
}

function updateStatusBadge(type, label) {
  const badge = el('status-badge');
  badge.textContent = label;
  badge.className = `status-badge status-${type}`;
}

/* ═══════════════════════════════════════════════════
   MAIN LOAD
═══════════════════════════════════════════════════ */
async function loadDashboard() {
  try {
    setLoadingState(true);

    const params = buildQueryParams();

    // Fetch all data in parallel
    const [scoresRes, trendRes] = await Promise.allSettled([
      apiFetch(`/flow-scores${params}`),
      apiFetch(`/trend?since=${AppState.since}&until=${AppState.until}`),
    ]);

    // Process flow scores
    if (scoresRes.status === 'fulfilled' && scoresRes.value.ok) {
      processFlowScores(scoresRes.value.data);
    } else {
      useMockFlowData();
    }

    // Process trend
    if (trendRes.status === 'fulfilled' && trendRes.value.ok) {
      renderTrendChart(trendRes.value.data.trend);
    } else {
      renderTrendChart(generateMockTrend());
    }

    setLoadingState(false);
  } catch (err) {
    console.warn('Dashboard load error:', err);
    useMockFlowData();
    renderTrendChart(generateMockTrend());
    setLoadingState(false);
  }
}

function buildQueryParams() {
  const parts = [];
  if (AppState.developer) parts.push(`developer=${encodeURIComponent(AppState.developer)}`);
  if (AppState.since)     parts.push(`since=${AppState.since}`);
  if (AppState.until)     parts.push(`until=${AppState.until}`);
  return parts.length ? `?${parts.join('&')}` : '';
}

/* ─── Process API flow scores response ─── */
function processFlowScores(data) {
  AppState.flowScores = data.scores || [];
  AppState.developers = data.developers || [];

  // Populate developer dropdown
  populateDevDropdown(data.developers);

  // Render summary stats
  renderSummaryStats(data.summary);

  // Render heatmap
  renderHeatmap(AppState.flowScores);

  // Render developer breakdown
  renderDevBreakdown();

  // Render anomaly log
  renderAnomalyLog(AppState.flowScores.filter(s => s.anomaly_label === -1));
}

/* ─── Populate developer dropdown ─── */
function populateDevDropdown(devs) {
  const sel = el('dev-select');
  const current = sel.value;
  sel.innerHTML = '<option value="">All Developers</option>';
  (devs || []).forEach(d => {
    const opt = document.createElement('option');
    opt.value = d;
    opt.textContent = d.length > 28 ? d.slice(0, 25) + '…' : d;
    sel.appendChild(opt);
  });
  if (current) sel.value = current;
}

/* ═══════════════════════════════════════════════════
   SUMMARY STATS
═══════════════════════════════════════════════════ */
function renderSummaryStats(summary) {
  if (!summary) return;

  // Flow score
  const score = summary.mean_flow_score?.toFixed(1) ?? '—';
  el('stat-flow-score').textContent = score;
  el('stat-flow-score').style.color = scoreColor(parseFloat(score));

  // Anomaly count
  setText('stat-anomaly-count', summary.anomaly_count ?? '—');

  // Worst developer
  if (summary.worst_day) {
    setText('stat-worst-dev', summary.worst_day.developer || 'N/A');
    setText('stat-worst-score', `Score: ${summary.worst_day.score?.toFixed(0) ?? '?'}/100`);
  }

  // Top recommendation (static derivation from worst feature)
  setText('stat-top-rec', deriveTopRecommendation(AppState.flowScores));
}

function scoreColor(score) {
  if (score >= 80) return '#10b981';
  if (score >= 60) return '#22c55e';
  if (score >= 40) return '#f97316';
  return '#ef4444';
}

function deriveTopRecommendation(scores) {
  const anomalies = scores.filter(s => s.anomaly_label === -1);
  if (!anomalies.length) return 'No anomalies detected — great flow!';
  return `${anomalies.length} disruption(s) found. Review PR review SLA and build time targets.`;
}

/* ═══════════════════════════════════════════════════
   HEATMAP (delegates to heatmap.js)
═══════════════════════════════════════════════════ */
function renderHeatmap(scores) {
  el('heatmap-loading').classList.add('hidden');
  el('heatmap-grid').classList.remove('hidden');
  FlowHeatmap.render('heatmap-grid', scores, AppState.since, AppState.until, onHeatmapCellClick);
}

function rerenderHeatmap() {
  const filtered = AppState.developer
    ? AppState.flowScores.filter(s => s.developer === AppState.developer)
    : AppState.flowScores;
  renderHeatmap(filtered);
}

function onHeatmapCellClick(date, developer, score) {
  // Open SHAP panel
  ShapPanel.open(date, developer, score);

  // Pre-populate simulator context
  SimulatorModule.setContext(developer, date, score);
}

/* ═══════════════════════════════════════════════════
   TREND CHART
═══════════════════════════════════════════════════ */
function renderTrendChart(trendData) {
  const canvas = el('trend-chart');
  const loadingEl = el('trend-loading');

  if (!trendData || !trendData.length) {
    if (loadingEl) loadingEl.textContent = 'No trend data available';
    return;
  }

  if (loadingEl) loadingEl.classList.add('hidden');

  const labels = trendData.map(d => formatDateLabel(d.date));
  const values = trendData.map(d => d.rolling_avg_score);
  const anomalyCounts = trendData.map(d => d.anomaly_count || 0);

  // Destroy previous chart
  if (AppState.trendChart) {
    AppState.trendChart.destroy();
    AppState.trendChart = null;
  }

  AppState.trendChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Rolling 7-Day Flow Score',
          data: values,
          borderColor: '#00e5ff',
          backgroundColor: createGradient(canvas, '#00e5ff'),
          borderWidth: 2,
          pointRadius: anomalyCounts.map(c => c > 0 ? 5 : 2),
          pointBackgroundColor: anomalyCounts.map(c => c > 0 ? '#ef4444' : '#00e5ff'),
          pointBorderColor: anomalyCounts.map(c => c > 0 ? '#ef4444' : '#00e5ff'),
          tension: 0.4,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#161822',
          borderColor: '#1e2130',
          borderWidth: 1,
          titleColor: '#e8eaf0',
          bodyColor: '#6b7280',
          padding: 10,
          callbacks: {
            label: ctx => {
              const ac = anomalyCounts[ctx.dataIndex];
              const lines = [`Flow Score: ${ctx.parsed.y?.toFixed(1)}/100`];
              if (ac > 0) lines.push(`⚠️ ${ac} anomaly${ac > 1 ? 's' : ''} on this day`);
              return lines;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: '#1e2130' },
          ticks: {
            color: '#6b7280',
            font: { family: 'monospace', size: 10 },
            maxTicksLimit: 12,
          },
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: '#1e2130' },
          ticks: {
            color: '#6b7280',
            font: { family: 'monospace', size: 10 },
            callback: v => `${v}`,
          },
        },
      },
    },
  });
}

function createGradient(canvas, color) {
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.parentElement.clientHeight || 200);
  gradient.addColorStop(0, color + '33');
  gradient.addColorStop(1, color + '00');
  return gradient;
}

function formatDateLabel(dateStr) {
  if (!dateStr) return '';
  const d = new Date(dateStr);
  return `${d.toLocaleString('en', { month: 'short' })} ${d.getDate()}`;
}

/* ═══════════════════════════════════════════════════
   DEVELOPER BREAKDOWN
═══════════════════════════════════════════════════ */
function renderDevBreakdown() {
  const container = el('dev-breakdown');
  const scores = AppState.flowScores;

  if (!scores.length) {
    container.innerHTML = '<div class="text-[#6b7280] text-sm text-center py-8">No data loaded</div>';
    return;
  }

  // Compute per-developer average
  const devMap = {};
  scores.forEach(s => {
    if (!devMap[s.developer]) devMap[s.developer] = { total: 0, count: 0 };
    devMap[s.developer].total += s.flow_score;
    devMap[s.developer].count += 1;
  });

  const devAverages = Object.entries(devMap)
    .map(([dev, stats]) => ({ dev, avg: stats.total / stats.count }))
    .sort((a, b) => b.avg - a.avg);

  container.innerHTML = '';
  devAverages.forEach(({ dev, avg }) => {
    const row = document.createElement('div');
    row.className = 'dev-bar-row fade-in';
    row.innerHTML = `
      <div class="dev-bar-name" title="${dev}">${shortenEmail(dev)}</div>
      <div class="dev-bar-outer">
        <div class="dev-bar-inner" style="width:${avg}%"></div>
      </div>
      <div class="dev-bar-score">${avg.toFixed(0)}</div>
    `;
    row.addEventListener('click', () => {
      const sel = el('dev-select');
      sel.value = dev;
      AppState.developer = dev;
      rerenderHeatmap();
    });
    container.appendChild(row);
  });
}

/* ═══════════════════════════════════════════════════
   ANOMALY LOG
═══════════════════════════════════════════════════ */
function renderAnomalyLog(anomalies) {
  const container = el('anomaly-log');

  if (!anomalies.length) {
    container.innerHTML = '<div class="text-[#6b7280] text-sm text-center py-8 card-body">No anomalies detected ✓</div>';
    return;
  }

  const sorted = [...anomalies].sort((a, b) => a.flow_score - b.flow_score).slice(0, 20);

  container.innerHTML = '<div class="card-body space-y-2">' +
    sorted.map(a => `
      <div class="anomaly-item fade-in" data-date="${a.date}" data-dev="${a.developer}" data-score="${a.flow_score}">
        <div class="flex-1 min-w-0">
          <div class="dev">${shortenEmail(a.developer)}</div>
          <div class="date">${a.date} · ${a.daily_commit_count ?? '?'} commits</div>
        </div>
        <div class="score-chip">${a.flow_score?.toFixed(0) ?? '?'}/100</div>
      </div>
    `).join('') +
  '</div>';

  // Wire click handlers
  container.querySelectorAll('.anomaly-item').forEach(item => {
    item.addEventListener('click', () => {
      const { date, dev, score } = item.dataset;
      ShapPanel.open(date, dev, parseFloat(score));
    });
  });
}

/* ═══════════════════════════════════════════════════
   OPEN SIMULATOR
═══════════════════════════════════════════════════ */
function openSimulator() {
  SimulatorModule.open();
}

/* ═══════════════════════════════════════════════════
   LOADING STATE
═══════════════════════════════════════════════════ */
function setLoadingState(loading) {
  const btn = el('btn-analyse');
  if (loading) {
    btn.textContent = '⏳ Analysing...';
    btn.disabled = true;
    el('heatmap-loading').classList.remove('hidden');
    el('heatmap-grid').classList.add('hidden');
  } else {
    btn.innerHTML = '<span class="mr-2">⚡</span> Re-analyse';
    btn.disabled = false;
  }
}

/* ═══════════════════════════════════════════════════
   MOCK / FALLBACK DATA
═══════════════════════════════════════════════════ */
function showFallback() {
  setText('header-subtitle', 'Demo mode — API not connected');
  useMockFlowData();
  renderTrendChart(generateMockTrend());
}

function useMockFlowData() {
  const scores = generateMockScores();
  processFlowScores({
    scores,
    developers: [...new Set(scores.map(s => s.developer))],
    summary: {
      mean_flow_score: 64.2,
      anomaly_count: scores.filter(s => s.anomaly_label === -1).length,
      worst_day: scores.sort((a, b) => a.flow_score - b.flow_score)[0],
    },
  });
}

function generateMockScores() {
  const developers = [
    'alice@example.com',
    'bob@example.com',
    'carol@example.com',
  ];

  const scores = [];
  const today = new Date();

  for (let i = 90; i >= 0; i--) {
    const date = new Date();
    date.setDate(today.getDate() - i);
    const dateStr = date.toISOString().slice(0, 10);

    developers.forEach(dev => {
      // Skip weekends for realism
      if (date.getDay() === 0 || date.getDay() === 6) return;

      // Create some anomaly clusters (simulate release weeks)
      const isAnomaly = (i >= 20 && i <= 25) || (i >= 50 && i <= 53) || Math.random() < 0.08;
      const base = isAnomaly ? 20 + Math.random() * 25 : 60 + Math.random() * 35;
      const score = Math.max(5, Math.min(98, base + (Math.random() - 0.5) * 15));

      scores.push({
        date: dateStr,
        developer: dev,
        flow_score: score,
        anomaly_label: isAnomaly ? -1 : 1,
        anomaly_score: isAnomaly ? -0.3 - Math.random() * 0.2 : -0.05,
        daily_commit_count: Math.floor(3 + Math.random() * 10),
        session_duration_minutes: Math.floor(60 + Math.random() * 300),
      });
    });
  }

  return scores;
}

function generateMockTrend() {
  const trend = [];
  const today = new Date();

  for (let i = 90; i >= 0; i--) {
    const date = new Date();
    date.setDate(today.getDate() - i);

    // Simulate dips during release periods
    const isLow = (i >= 20 && i <= 25) || (i >= 50 && i <= 53);
    const base = isLow ? 35 + Math.random() * 15 : 62 + Math.random() * 20;

    trend.push({
      date: date.toISOString().slice(0, 10),
      rolling_avg_score: Math.max(10, Math.min(95, base)),
      anomaly_count: isLow ? Math.floor(1 + Math.random() * 3) : 0,
    });
  }

  return trend;
}

/* ═══════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════ */
function shortenEmail(email) {
  if (!email) return '—';
  const local = email.split('@')[0];
  return local.length > 18 ? local.slice(0, 16) + '…' : local;
}

/* Expose app state for sub-modules */
window.AppState = AppState;
window.apiFetch = apiFetch;
window.shortenEmail = shortenEmail;