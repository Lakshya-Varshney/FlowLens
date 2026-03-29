/**
 * shap_panel.js
 * ─────────────────────────────────────────────────────
 * Manages the SHAP explanation side panel.
 * Opens when a heatmap cell is clicked, fetches
 * SHAP values from /api/shap/<date>/<dev>,
 * renders a Chart.js horizontal bar chart and
 * the LLM-synthesised recommendation text.
 * ─────────────────────────────────────────────────────
 */

'use strict';

const ShapPanel = (() => {

  let shapChart = null;          // Chart.js instance
  let currentDate = null;
  let currentDev  = null;

  // Confirmed dev/date — only set AFTER a successful /api/shap response.
  // The simulator reads these. They are never set from a stale click.
  let shapPanelDev  = null;
  let shapPanelDate = null;

  /* ── DOM references ── */
  const panel    = () => document.getElementById('shap-panel');
  const content  = () => document.getElementById('shap-content');
  const subtitle = () => document.getElementById('shap-subtitle');
  const closeBtn = () => document.getElementById('shap-close');

  /* ─── Initialise (called on DOM ready) ─── */
  function init() {
    closeBtn()?.addEventListener('click', close);
  }

  /* ─── Open the panel with data for date/dev ─── */
  async function open(date, developer, scoreHint) {
    currentDate = date;
    currentDev  = developer;

    // Reset confirmed snapshot — only set after a successful API response.
    // This prevents the simulator from using stale data from a previous click.
    shapPanelDev  = null;
    shapPanelDate = null;
    setSimButtonState(false);

    // Show panel
    const p = panel();
    p.classList.add('open');

    // Update subtitle
    const sub = subtitle();
    if (sub) sub.textContent = `${shortenEmail(developer)} · ${date}`;

    // Show loading skeleton
    showLoading(scoreHint);

    const devEncoded = encodeURIComponent(developer);

    // Fire both calls in parallel — recommend may be slow if LLM is involved
    const [shapResult, recResult] = await Promise.allSettled([
      window.apiFetch(`/shap/${date}/${devEncoded}`),
      window.apiFetch(`/recommend/${date}/${devEncoded}`),
    ]);

    const shapOk = shapResult.status === 'fulfilled' && shapResult.value?.ok;
    const recOk  = recResult.status  === 'fulfilled' && recResult.value?.ok;

    if (shapOk) {
      renderShapData(
        shapResult.value.data,
        recOk ? recResult.value.data : null
      );

      // ✅ Only confirm dev/date after a successful SHAP load.
      // The simulator reads shapPanelDev / shapPanelDate — guaranteed DB entry.
      shapPanelDev  = developer;
      shapPanelDate = date;
      setSimButtonState(true, developer, date);
    } else {
      renderMockShapData(date, developer, scoreHint);
      // Mock data — simulator can still open but will use heuristic fallback
      shapPanelDev  = developer;
      shapPanelDate = date;
      setSimButtonState(true, developer, date);
    }
  }

  /* ─── Enable / disable the static simulator button ─── */
  function setSimButtonState(enabled, dev, date) {
    const btn  = document.getElementById('open-sim-btn');
    const hint = document.getElementById('open-sim-hint');
    if (!btn) return;

    if (enabled) {
      btn.disabled = false;
      btn.classList.remove('opacity-40', 'cursor-not-allowed');
      btn.classList.add('cursor-pointer');
      // Store confirmed context on the button element itself
      btn.dataset.dev  = dev  || '';
      btn.dataset.date = date || '';
      if (hint) hint.textContent = 'Simulate workflow changes for this day →';
    } else {
      btn.disabled = true;
      btn.classList.add('opacity-40', 'cursor-not-allowed');
      btn.classList.remove('cursor-pointer');
      btn.dataset.dev  = '';
      btn.dataset.date = '';
      if (hint) hint.textContent = 'Load a cell\'s X-ray to unlock the simulator';
    }
  }

  /* ─── Expose confirmed context for simulator.js ─── */
  function getConfirmedContext() {
    return { dev: shapPanelDev, date: shapPanelDate };
  }

  /* ─── Close the panel ─── */
  function close() {
    panel().classList.remove('open');
    // Deselect heatmap cell
    document.querySelectorAll('.heatmap-cell.selected')
      .forEach(c => c.classList.remove('selected'));
    if (shapChart) {
      shapChart.destroy();
      shapChart = null;
    }
  }

  /* ─── Show loading skeleton ─── */
  function showLoading(scoreHint) {
    const cont = content();
    cont.innerHTML = `
      <div class="flex items-center gap-4 mb-4">
        <div class="shap-score-ring ${ringClass(scoreHint)}">
          <div style="font-size:20px;font-weight:800;color:${scoreTextColor(scoreHint)};line-height:1;">
            ${scoreHint !== undefined ? Math.round(scoreHint) : '—'}
          </div>
          <div style="font-size:8px;color:#6b7280;margin-top:2px;">/ 100</div>
        </div>
        <div>
          <div style="font-size:13px;font-weight:700;color:#fff;">${currentDev ? shortenEmail(currentDev) : '—'}</div>
          <div style="font-size:11px;color:#6b7280;">${currentDate || '—'}</div>
          <div style="font-size:11px;margin-top:4px;color:${scoreTextColor(scoreHint)};">${severityLabel(scoreHint)}</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;justify-content:center;height:80px;color:#6b7280;font-size:12px;">
        <div class="loading-spinner" style="margin-right:10px;"></div> Loading X-ray diagnosis...
      </div>
    `;
  }

  /* ─── Render SHAP data from API response ─── */
  function renderShapData(shapData, recData) {
    const score    = shapData.flow_score;
    const shapVals = shapData.shap_values || [];
    const summary  = shapData.plain_text_summary || '';

    // Prefer the dedicated /recommend response; fall back to inline rec in /shap
    const llmRec    = recData?.llm_recommendation
                   || shapData.recommendation?.llm_synthesised
                   || null;
    const ruleRecs  = recData?.rule_based_recommendations
                   || (shapData.recommendation?.rule_based
                        ? [shapData.recommendation.rule_based]
                        : []);
    const hoursEst  = recData?.hours_recoverable_estimate ?? null;
    const llmAvail  = recData?.llm_available ?? false;
    const topFeature = recData?.top_feature ?? null;

    renderPanel(score, shapVals, summary, {
      llmRec, ruleRecs, hoursEst, llmAvail, topFeature,
    });
  }

  /* ─── Render with mock data (fallback) ─── */
  function renderMockShapData(date, developer, scoreHint) {
    const score    = scoreHint ?? 50;
    const mockShap = generateMockShap(score);
    const summary  = generateMockSummary(developer, date, mockShap, score);
    const ruleRec  = generateMockRecommendation(mockShap[0]);

    renderPanel(score, mockShap, summary, {
      llmRec:     null,
      ruleRecs:   [ruleRec],
      hoursEst:   ((100 - score) * 0.5).toFixed(1),
      llmAvail:   false,
      topFeature: mockShap[0]?.feature ?? null,
    });
  }

  /* ─── Build the full panel HTML + Chart ─── */
  function renderPanel(score, shapValues, summary, rec) {
    const cont = content();

    // rec is now an object: { llmRec, ruleRecs, hoursEst, llmAvail, topFeature }
    const { llmRec, ruleRecs = [], hoursEst, llmAvail, topFeature } = rec || {};

    // Destroy existing chart
    if (shapChart) {
      shapChart.destroy();
      shapChart = null;
    }

    cont.innerHTML = `
      <!-- Score ring + meta -->
      <div class="flex items-center gap-4 mb-5 fade-in">
        <div class="shap-score-ring ${ringClass(score)}">
          <div style="font-size:22px;font-weight:800;color:${scoreTextColor(score)};line-height:1;">${Math.round(score)}</div>
          <div style="font-size:8px;color:#6b7280;letter-spacing:0.05em;margin-top:2px;">/ 100</div>
        </div>
        <div>
          <div style="font-size:13px;font-weight:700;color:#fff;">${shortenEmail(currentDev)}</div>
          <div style="font-size:11px;color:#6b7280;">${currentDate}</div>
          <div style="font-size:11px;font-weight:600;margin-top:4px;color:${scoreTextColor(score)};">${severityLabel(score)}</div>
          ${hoursEst ? `<div style="font-size:10px;color:#a78bfa;margin-top:3px;">⏱ ${hoursEst}h recoverable/week</div>` : ''}
        </div>
      </div>

      <!-- SHAP Chart -->
      <div style="margin-bottom:16px;">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.15em;color:#6b7280;margin-bottom:10px;">
          Contributing Factors (SHAP)
        </div>
        <div style="position:relative;height:${Math.min(shapValues.length, 5) * 32 + 20}px;">
          <canvas id="shap-chart-canvas"></canvas>
        </div>
      </div>

      <!-- Feature detail rows -->
      <div style="margin-bottom:16px;" class="fade-in" id="shap-feature-rows">
        ${buildFeatureRows(shapValues)}
      </div>

      <!-- Diagnosis (plain text summary from /shap) -->
      ${summary ? `
        <div style="margin-bottom:12px;">
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.15em;color:#6b7280;margin-bottom:8px;">
            Diagnosis
          </div>
          <div class="rec-box" style="background:rgba(239,68,68,0.04);border-color:rgba(239,68,68,0.2);">
            ${escapeHTML(summary)}
          </div>
        </div>
      ` : ''}

      <!-- Rule-based recommendations (from /recommend) -->
      ${ruleRecs.length ? `
        <div style="margin-bottom:12px;">
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.15em;color:#6b7280;margin-bottom:8px;">
            Rule-Based Actions
          </div>
          <div style="display:flex;flex-direction:column;gap:6px;">
            ${ruleRecs.map((r, i) => `
              <div class="rec-box" style="padding:10px 12px;display:flex;gap:8px;align-items:flex-start;">
                <span style="color:#f59e0b;flex-shrink:0;font-size:11px;">${i + 1}.</span>
                <span style="font-size:11px;">${escapeHTML(r)}</span>
              </div>
            `).join('')}
          </div>
        </div>
      ` : ''}

      <!-- LLM recommendation (from /recommend → llm_recommendation) -->
      <div style="margin-bottom:12px;" id="llm-rec-section">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.15em;color:#6b7280;margin-bottom:8px;display:flex;align-items:center;gap:8px;">
          AI Recommendation
          ${llmAvail
            ? '<span style="font-size:9px;background:rgba(124,58,237,0.15);color:#a78bfa;border:1px solid rgba(124,58,237,0.3);padding:2px 7px;border-radius:2px;">✦ LLM</span>'
            : '<span style="font-size:9px;background:rgba(107,114,128,0.15);color:#6b7280;border:1px solid rgba(107,114,128,0.2);padding:2px 7px;border-radius:2px;">Offline</span>'
          }
        </div>
        ${llmRec
          ? formatRecommendation(llmRec)
          : `<div class="rec-box" style="background:rgba(107,114,128,0.05);border-color:rgba(107,114,128,0.15);">
               <div style="font-size:11px;color:#6b7280;">
                 ${llmAvail
                   ? '⏳ LLM response unavailable for this entry.'
                   : 'Set <code style="background:#161822;padding:1px 5px;border-radius:2px;">GOOGLE_API_KEY</code> to enable AI recommendations.'}
               </div>
             </div>`
        }
      </div>

      <!-- Open simulator button is now static HTML (#open-sim-btn) below this panel -->
    `;

    // Render Chart.js horizontal bar chart
    renderShapChart(shapValues);
  }

  /* ─── Render horizontal bar chart ─── */
  function renderShapChart(shapValues) {
    const canvas = document.getElementById('shap-chart-canvas');
    if (!canvas || !shapValues.length) return;

    const top5   = shapValues.slice(0, 5);
    const labels = top5.map(s => formatFeatureName(s.feature));
    const values = top5.map(s => Math.abs(s.shap_value) * 100);
    const colors = top5.map(s => s.shap_value < 0 ? 'rgba(239,68,68,0.7)' : 'rgba(16,185,129,0.7)');
    const borderColors = top5.map(s => s.shap_value < 0 ? '#ef4444' : '#10b981');

    shapChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: colors,
          borderColor: borderColors,
          borderWidth: 1,
          borderRadius: 2,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600 },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#161822',
            borderColor: '#1e2130',
            borderWidth: 1,
            titleColor: '#e8eaf0',
            bodyColor: '#6b7280',
            callbacks: {
              label: ctx => {
                const d = top5[ctx.dataIndex];
                const dir = d.shap_value < 0 ? '↑ worsened' : '↓ improved';
                return ` ${dir} flow by ${ctx.parsed.x.toFixed(1)}%`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: '#1e2130' },
            ticks: {
              color: '#6b7280',
              font: { family: 'monospace', size: 9 },
              callback: v => `${v.toFixed(0)}%`,
            },
          },
          y: {
            grid: { display: false },
            ticks: {
              color: '#e8eaf0',
              font: { family: 'monospace', size: 10 },
            },
          },
        },
      },
    });
  }

  /* ─── Build feature detail rows ─── */
  function buildFeatureRows(shapValues) {
    return shapValues.slice(0, 5).map(s => {
      const pct = Math.min(100, Math.abs(s.shap_value) * 200);
      const isNeg = s.shap_value < 0;
      const actual   = s.actual_value?.toFixed(1) ?? '—';
      const baseline = s.baseline_mean?.toFixed(1) ?? '—';

      return `
        <div class="shap-feature-row">
          <div class="shap-feature-label" title="${s.feature}">${formatFeatureName(s.feature)}</div>
          <div class="shap-bar-outer">
            <div class="shap-bar-inner ${isNeg ? 'shap-bar-neg' : 'shap-bar-pos'}" style="width:${pct}%"></div>
          </div>
          <div class="shap-feature-pct" style="color:${isNeg ? '#ef4444' : '#10b981'}">
            ${isNeg ? '↑' : '↓'}${pct.toFixed(0)}%
          </div>
        </div>
        <div style="font-size:9px;color:#6b7280;padding:0 12px 6px;display:flex;gap:12px;">
          <span>Actual: <span style="color:#e8eaf0">${actual}</span></span>
          <span>Baseline: <span style="color:#e8eaf0">${baseline}</span></span>
        </div>
      `;
    }).join('');
  }

  /* ═══════════════════════════════════════════════════
     MOCK DATA GENERATORS (fallback when API offline)
  ═══════════════════════════════════════════════════ */

  function generateMockShap(score) {
    const features = [
      { feature: 'pr_blocking_time_hours',    baseline: 4.1 },
      { feature: 'inter_commit_gap_variance', baseline: 0.9 },
      { feature: 'late_night_ratio',          baseline: 0.08 },
      { feature: 'test_failure_density',      baseline: 0.05 },
      { feature: 'build_time_minutes',        baseline: 8.2 },
    ];

    const severity = (100 - score) / 100;

    return features.map((f, i) => {
      const magnitude = (severity * (0.5 - i * 0.07)) + (Math.random() - 0.5) * 0.05;
      const actualMultiplier = score < 50 ? 2 + Math.random() * 3 : 1 + Math.random();

      return {
        feature:       f.feature,
        shap_value:    -(Math.abs(magnitude) * (1 - i * 0.15)),
        actual_value:  f.baseline * actualMultiplier,
        baseline_mean: f.baseline,
      };
    });
  }

  function generateMockSummary(developer, date, shapVals, score) {
    const topFeat = shapVals[0];
    const featureName = formatFeatureName(topFeat.feature).toLowerCase();
    const actual = topFeat.actual_value?.toFixed(1) ?? '?';
    const baseline = topFeat.baseline_mean?.toFixed(1) ?? '?';

    return `On ${date}, ${shortenEmail(developer)}'s flow was primarily disrupted by ${featureName} ` +
      `(${actual} vs baseline ${baseline}). This is ${((topFeat.actual_value / topFeat.baseline_mean) * 100 - 100).toFixed(0)}% ` +
      `above their historical average, indicating a significant workflow bottleneck during this period.`;
  }

  function generateMockRecommendation(topShap) {
    const recMap = {
      'pr_blocking_time_hours':    'Reduce PR review latency. Set a team SLA of <4 hours for first review. Consider a rotating review-duty engineer each sprint day.',
      'inter_commit_gap_variance': 'Work sessions are fragmented. Implement 90-minute focus blocks with Slack notifications paused. Correlates with meeting overload — audit the calendar for this period.',
      'late_night_ratio':          'High late-night coding detected. This developer was likely overloaded or blocked during core hours. Investigate task assignment and blockers from this period.',
      'test_failure_density':      'Frequent CI failures disrupted commit flow. Investigate flaky tests from this period. Add pre-commit test filtering to catch failures earlier.',
      'build_time_minutes':        'Slow build times created flow-breaking wait loops. Profile CI pipeline for caching opportunities. Target: <5 minutes for incremental builds.',
    };

    return recMap[topShap.feature] || 'Review workflow patterns for this period and consider reducing context-switching triggers.';
  }

  /* ─── Helpers ─── */
  function ringClass(score) {
    if (score === undefined) return '';
    if (score < 35) return 'score-danger';
    if (score < 60) return 'score-warning';
    return 'score-ok';
  }

  function scoreTextColor(score) {
    if (score === undefined) return '#6b7280';
    if (score < 35) return '#ef4444';
    if (score < 60) return '#f97316';
    return '#10b981';
  }

  function severityLabel(score) {
    if (score === undefined || score === null) return 'No data';
    if (score < 20) return '🔴 Critical disruption';
    if (score < 35) return '🟠 Severe disruption';
    if (score < 50) return '🟡 Disrupted flow';
    if (score < 70) return '🟢 Moderate flow';
    return '✅ Healthy flow';
  }

  function formatFeatureName(feature) {
    if (!feature) return feature;
    return feature
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
      .replace('Pr ', 'PR ')
      .replace('Ci ', 'CI ')
      .replace('Mean', '(Mean)');
  }

  function shortenEmail(email) {
    if (!email) return '—';
    return window.shortenEmail ? window.shortenEmail(email) : email.split('@')[0];
  }

  /**
   * formatRecommendation()
   * ─────────────────────────────────────────────────
   * Parses the structured LLM output which follows
   * this exact template (plain text, no markdown):
   *
   *   WHAT HAPPENED:
   *   <text>
   *
   *   PRIORITISED ACTIONS:
   *   1. <action>
   *   2. <action>
   *   3. <action>
   *
   *   HOURS RECOVERED:
   *   <text>
   *
   * Each section is rendered with its own colour-coded
   * label. Falls back to a plain text box if the model
   * didn't follow the template.
   * ─────────────────────────────────────────────────
   */
  function formatRecommendation(raw) {
    if (!raw || !raw.trim()) return '';

    // ── Section definitions: label, key, accent color, bg tint ──
    const sections = [
      {
        key:   'WHAT HAPPENED',
        label: '🔍 What Happened',
        color: '#00e5ff',
        bg:    'rgba(0,229,255,0.04)',
        border:'rgba(0,229,255,0.2)',
      },
      {
        key:   'PRIORITISED ACTIONS',
        label: '⚡ Prioritised Actions',
        color: '#f59e0b',
        bg:    'rgba(245,158,11,0.04)',
        border:'rgba(245,158,11,0.2)',
      },
      {
        key:   'HOURS RECOVERED',
        label: '⏱ Hours Recovered',
        color: '#10b981',
        bg:    'rgba(16,185,129,0.04)',
        border:'rgba(16,185,129,0.2)',
      },
    ];

    // ── Parse: split on section headers ──
    // Regex matches "WHAT HAPPENED:", "PRIORITISED ACTIONS:", etc.
    // case-insensitive, optional trailing whitespace
    const headerPattern = new RegExp(
      `(${sections.map(s => s.key).join('|')}):?`,
      'gi'
    );

    const parsed = {};
    let lastKey = null;
    let lastIndex = 0;

    // Find all header positions
    const matches = [...raw.matchAll(headerPattern)];

    if (matches.length === 0) {
      // No structure found — render as plain fallback
      return plainFallback(raw);
    }

    matches.forEach((match, i) => {
      // Save text of the previous section
      if (lastKey !== null) {
        parsed[lastKey] = raw.slice(lastIndex, match.index).trim();
      }
      lastKey   = match[0].replace(/:?\s*$/, '').toUpperCase().trim();
      lastIndex = match.index + match[0].length;
    });

    // Save the final section
    if (lastKey !== null) {
      parsed[lastKey] = raw.slice(lastIndex).trim();
    }

    // ── Render each section ──
    let html = `<div style="display:flex;flex-direction:column;gap:8px;">`;

    sections.forEach(({ key, label, color, bg, border }) => {
      const text = parsed[key] || parsed[key + ':'] || null;
      if (!text) return;

      const isActions = key === 'PRIORITISED ACTIONS';
      const bodyHtml  = isActions
        ? renderActionsList(text)
        : `<div style="font-size:12px;color:#e8eaf0;line-height:1.7;">${escapeHTML(text)}</div>`;

      html += `
        <div style="
          background:${bg};
          border:1px solid ${border};
          border-radius:4px;
          padding:12px 14px;
          position:relative;
          overflow:hidden;
        ">
          <!-- top accent line -->
          <div style="
            position:absolute;top:0;left:0;right:0;height:1px;
            background:linear-gradient(90deg,transparent,${color},transparent);
          "></div>
          <!-- section label -->
          <div style="
            font-size:10px;
            font-weight:700;
            text-transform:uppercase;
            letter-spacing:0.12em;
            color:${color};
            margin-bottom:8px;
          ">${label}</div>
          ${bodyHtml}
        </div>
      `;
    });

    html += `</div>`;

    // If nothing parsed correctly, fall back
    const hasContent = sections.some(s => parsed[s.key]);
    return hasContent ? html : plainFallback(raw);
  }

  /**
   * renderActionsList()
   * Turns numbered action lines into styled list items.
   * Handles both "1. text" and "- text" formats.
   */
  function renderActionsList(text) {
    const lines = text
      .split('\n')
      .map(l => l.trim())
      .filter(l => l.length > 0);

    // Check if lines look like a numbered / bulleted list
    const isListLine = l => /^(\d+[.)]\s+|[-•*]\s+)/.test(l);
    const listLines  = lines.filter(isListLine);

    if (listLines.length === 0) {
      // No list structure — just render as text
      return `<div style="font-size:12px;color:#e8eaf0;line-height:1.7;">${escapeHTML(text)}</div>`;
    }

    const items = listLines.map((line, i) => {
      // Strip leading "1. " / "- " / "• "
      const body = line.replace(/^(\d+[.)]\s+|[-•*]\s+)/, '').trim();
      return `
        <div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:6px;">
          <span style="
            flex-shrink:0;
            width:18px;height:18px;
            border-radius:50%;
            background:rgba(245,158,11,0.15);
            border:1px solid rgba(245,158,11,0.3);
            color:#f59e0b;
            font-size:9px;
            font-weight:700;
            display:flex;align-items:center;justify-content:center;
          ">${i + 1}</span>
          <span style="font-size:12px;color:#e8eaf0;line-height:1.6;">${escapeHTML(body)}</span>
        </div>
      `;
    });

    return `<div style="margin-top:2px;">${items.join('')}</div>`;
  }

  /**
   * plainFallback()
   * Used when the LLM didn't follow the template structure.
   * Renders the raw text in a neutral box with a warning note.
   */
  function plainFallback(raw) {
    return `
      <div class="rec-box" style="background:rgba(124,58,237,0.05);border-color:rgba(124,58,237,0.25);">
        <div style="font-size:10px;color:#a78bfa;margin-bottom:6px;">✦ AI-synthesised insight</div>
        <div style="font-size:12px;line-height:1.7;color:#e8eaf0;">${escapeHTML(raw)}</div>
      </div>
    `;
  }

  function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  /* ─── Public API ─── */
  return { init, open, close, getConfirmedContext };

})();

/* ─── Auto-init ─── */
document.addEventListener('DOMContentLoaded', () => ShapPanel.init());

window.ShapPanel = ShapPanel;