/**
 * heatmap.js
 * ─────────────────────────────────────────────────────
 * Renders a GitHub-style contribution heatmap using
 * plain DOM elements. Each cell represents one
 * developer-day. Clicking opens the SHAP panel.
 * ─────────────────────────────────────────────────────
 */

'use strict';

const FlowHeatmap = (() => {

  /* ── Score → CSS class mapping ── */
  function scoreClass(score) {
    if (score === null || score === undefined) return 'score-empty';
    if (score < 20)  return 'score-critical';
    if (score < 35)  return 'score-low';
    if (score < 50)  return 'score-med-low';
    if (score < 60)  return 'score-medium';
    if (score < 70)  return 'score-med-high';
    if (score < 85)  return 'score-high';
    return 'score-excellent';
  }

  /* ── Score → hex color for tooltip bar ── */
  function scoreHex(score) {
    if (score === null || score === undefined) return '#111318';
    if (score < 20)  return '#450a0a';
    if (score < 35)  return '#7f1d1d';
    if (score < 50)  return '#b45309';
    if (score < 60)  return '#f97316';
    if (score < 70)  return '#eab308';
    if (score < 85)  return '#22c55e';
    return '#10b981';
  }

  /* ── Score severity label ── */
  function severityLabel(score) {
    if (score === null) return 'No data';
    if (score < 20)  return '🔴 Critical';
    if (score < 35)  return '🟠 Severe';
    if (score < 50)  return '🟡 Disrupted';
    if (score < 70)  return '🟢 Moderate';
    return '✅ Good Flow';
  }

  /* ── Build a lookup map: "YYYY-MM-DD::email" → score object ── */
  function buildScoreMap(scores) {
    const map = new Map();
    scores.forEach(s => {
      // Per developer-day key — used when a specific dev is selected
      const key = `${s.date}::${s.developer}`;
      map.set(key, s);

      // ALL-devs aggregate per date — keeps the REAL developer email of the
      // worst-scoring entry so clicking the cell sends a valid email to the API.
      // The tooltip displays "Team (worst)" separately; the data object must
      // never contain a fake string like 'Team worst'.
      const dayKey = `${s.date}::ALL`;
      const existing = map.get(dayKey);
      if (!existing) {
        map.set(dayKey, { ...s });          // first entry — copy as-is
      } else if (s.flow_score != null && s.flow_score < (existing.flow_score ?? 100)) {
        map.set(dayKey, { ...s });          // lower score — replace, keep real developer
      }
    });
    return map;
  }

  /* ── Generate a list of dates from since→until ── */
  function generateDateRange(since, until) {
    const dates = [];
    const start = new Date(since || getDefaultSince());
    const end   = new Date(until || new Date().toISOString().slice(0, 10));

    // Align to the Monday of the start week
    const cursor = new Date(start);
    cursor.setDate(cursor.getDate() - cursor.getDay() + 1); // Monday

    while (cursor <= end) {
      dates.push(new Date(cursor));
      cursor.setDate(cursor.getDate() + 1);
    }

    return dates;
  }

  function getDefaultSince() {
    const d = new Date();
    d.setDate(d.getDate() - 90);
    return d.toISOString().slice(0, 10);
  }

  function formatDate(d) {
    return d.toISOString().slice(0, 10);
  }

  function formatDisplay(d) {
    return d.toLocaleDateString('en', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
  }

  /* ── Tooltip management ── */
  function showTooltip(x, y, content) {
    const tip = document.getElementById('heatmap-tooltip');
    if (!tip) return;
    tip.innerHTML = content;
    tip.classList.remove('hidden');

    // Position tooltip (avoid edge overflow)
    const margin = 12;
    const tw = 200;
    const th = 100;
    let left = x + margin;
    let top  = y - margin;

    if (left + tw > window.innerWidth)  left = x - tw - margin;
    if (top + th > window.innerHeight)  top = window.innerHeight - th - margin;
    if (top < 0) top = margin;

    tip.style.left = `${left}px`;
    tip.style.top  = `${top}px`;
  }

  function hideTooltip() {
    const tip = document.getElementById('heatmap-tooltip');
    if (tip) tip.classList.add('hidden');
  }

  /* ── Main render function ── */
  function render(containerId, scores, since, until, onCellClick) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    const isAllDevs = !window.AppState?.developer;
    const scoreMap  = buildScoreMap(scores);
    const dates     = generateDateRange(since, until);
    const lookupSuffix = isAllDevs ? 'ALL' : (window.AppState?.developer || 'ALL');

    /* ── Outer wrapper ── */
    const wrapper = document.createElement('div');
    wrapper.style.display    = 'flex';
    wrapper.style.gap        = '6px';
    wrapper.style.alignItems = 'flex-start';

    /* ── Day-of-week labels (left column) ── */
    const dayLabels = document.createElement('div');
    dayLabels.className = 'heatmap-day-labels';
    dayLabels.style.marginTop = '20px'; // align with first row of cells

    ['Mon', '', 'Wed', '', 'Fri', '', 'Sun'].forEach(label => {
      const lbl = document.createElement('div');
      lbl.className = 'heatmap-day-label';
      lbl.textContent = label;
      dayLabels.appendChild(lbl);
    });
    wrapper.appendChild(dayLabels);

    /* ── Grid area (weeks as columns) ── */
    const gridArea = document.createElement('div');
    gridArea.style.display  = 'flex';
    gridArea.style.flexDirection = 'column';
    gridArea.style.gap      = '0';

    /* -- Month labels row -- */
    const monthRow = document.createElement('div');
    monthRow.style.display = 'flex';
    monthRow.style.marginBottom = '4px';
    monthRow.style.height = '14px';

    /* -- Weeks container -- */
    const weeksRow = document.createElement('div');
    weeksRow.style.display = 'flex';
    weeksRow.style.gap     = '3px';

    /* Group dates into weeks (each week = 7 days Mon→Sun) */
    const weeks = [];
    for (let i = 0; i < dates.length; i += 7) {
      weeks.push(dates.slice(i, i + 7));
    }

    /* Track last rendered month for label placement */
    let lastMonthLabel = '';
    const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

    weeks.forEach((week, weekIdx) => {
      /* Month label spacer */
      const firstDate = week[0];
      const monthKey  = `${firstDate.getFullYear()}-${firstDate.getMonth()}`;
      const lbl = document.createElement('div');
      lbl.style.width    = '16px';
      lbl.style.fontSize = '9px';
      lbl.style.color    = '#6b7280';
      lbl.style.flexShrink = '0';

      if (monthKey !== lastMonthLabel) {
        lbl.textContent = monthNames[firstDate.getMonth()];
        lastMonthLabel  = monthKey;
      }
      monthRow.appendChild(lbl);

      /* Week column */
      const weekCol = document.createElement('div');
      weekCol.style.display       = 'flex';
      weekCol.style.flexDirection = 'column';
      weekCol.style.gap           = '3px';

      week.forEach(date => {
        const dateStr = formatDate(date);
        const key = `${dateStr}::${lookupSuffix}`;
        const data = scoreMap.get(key);

        const cell = document.createElement('div');
        cell.className = `heatmap-cell ${data ? scoreClass(data.flow_score) : 'score-empty'}`;
        cell.setAttribute('data-date', dateStr);
        cell.setAttribute('data-score', data?.flow_score ?? '');
        cell.setAttribute('data-dev',  data?.developer ?? '');

        /* Tooltip on hover */
        cell.addEventListener('mouseenter', e => {
          const score  = data?.flow_score;
          const dev    = isAllDevs ? 'Team (worst)' : (data?.developer ?? 'No data');
          const commits = data?.daily_commit_count ?? 0;

          showTooltip(e.clientX, e.clientY, `
            <div style="font-size:10px;color:#6b7280;margin-bottom:5px;">${formatDisplay(date)}</div>
            <div style="font-size:12px;font-weight:700;color:#fff;margin-bottom:4px;">
              ${score !== undefined ? `${score.toFixed(0)}/100` : 'No data'}
            </div>
            <div style="font-size:11px;color:${scoreHex(score)};margin-bottom:4px;">${severityLabel(score)}</div>
            <div style="font-size:10px;color:#6b7280;">${dev}</div>
            ${commits ? `<div style="font-size:10px;color:#6b7280;">${commits} commits</div>` : ''}
          `);
        });

        cell.addEventListener('mousemove', e => {
          const tip = document.getElementById('heatmap-tooltip');
          if (tip && !tip.classList.contains('hidden')) {
            showTooltip(e.clientX, e.clientY, tip.innerHTML);
          }
        });

        cell.addEventListener('mouseleave', () => hideTooltip());

        /* Click handler */
        if (data && onCellClick) {
          cell.addEventListener('click', e => {
            // Deselect previous
            container.querySelectorAll('.heatmap-cell.selected')
              .forEach(c => c.classList.remove('selected'));
            cell.classList.add('selected');

            hideTooltip();
            onCellClick(data.date, data.developer, data.flow_score);
          });
        }

        weekCol.appendChild(cell);
      });

      weeksRow.appendChild(weekCol);
    });

    gridArea.appendChild(monthRow);
    gridArea.appendChild(weeksRow);
    wrapper.appendChild(gridArea);

    container.appendChild(wrapper);
    container.classList.remove('hidden');
    container.classList.add('fade-in');
  }

  /* ── Public API ── */
  return { render };

})();

window.FlowHeatmap = FlowHeatmap;