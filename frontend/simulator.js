/**
 * simulator.js
 * ─────────────────────────────────────────────────────
 * Before/After Flow Score Simulator.
 * Sliders control workflow parameters.
 * POSTs to /api/simulate and displays an animated
 * score transition with recovered hours estimation.
 * ─────────────────────────────────────────────────────
 */

'use strict';

const SimulatorModule = (() => {

  /* ── State ── */
  let context = { developer: null, date: null, currentScore: null };
  let animFrame = null;

  /* ── DOM helpers ── */
  const el = id => document.getElementById(id);
  const setText = (id, text) => {
    const node = el(id);
    if (node) node.textContent = text;
  };

  /* ── Slider → label map ── */
  const sliders = [
    {
      id: 'slider-pr-time',
      valId: 'val-pr-time',
      format: v => `${v}h`,
      apiKey: 'pr_blocking_time_hours',
    },
    {
      id: 'slider-pr-size',
      valId: 'val-pr-size',
      format: v => `${v} lines`,
      apiKey: 'pr_size_lines',
    },
    {
      id: 'slider-build-time',
      valId: 'val-build-time',
      format: v => `${v} min`,
      apiKey: 'build_time_minutes',
    },
    {
      id: 'slider-focus',
      valId: 'val-focus',
      format: v => `${v} min`,
      apiKey: 'focus_block_minutes',
    },
  ];

  /* ═══════════════════════════════════════════════════
     INIT
  ═══════════════════════════════════════════════════ */
  function init() {
    // Wire slider live-update labels
    sliders.forEach(({ id, valId, format }) => {
      const slider = el(id);
      if (!slider) return;
      slider.addEventListener('input', () => {
        setText(valId, format(slider.value));
      });
    });

    // Run simulation button
    el('btn-run-simulation')?.addEventListener('click', runSimulation);

    // Reset button
    el('btn-clear-sim')?.addEventListener('click', resetSimulator);

    // Close button
    el('simulator-close')?.addEventListener('click', closeSimulator);

    // Close modal on overlay click
    el('simulator-modal')?.addEventListener('click', e => {
      if (e.target === el('simulator-modal')) closeSimulator();
    });

    // ESC key
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && !el('simulator-modal').classList.contains('hidden')) {
        closeSimulator();
      }
    });
  }

  /* ═══════════════════════════════════════════════════
     OPEN / CLOSE
  ═══════════════════════════════════════════════════ */
  function open() {
    el('simulator-modal').classList.remove('hidden');
    updateContextBar();
    updateCurrentScoreDisplay();
  }

  /**
   * openFromShap()
   * Called by the static #open-sim-btn in the SHAP panel.
   * Reads the confirmed dev/date from ShapPanel.getConfirmedContext()
   * — these are only set after a successful /api/shap response so we
   * know the entry exists in the database (no 404 on /api/simulate).
   */
  function openFromShap() {
    const { dev, date } = window.ShapPanel?.getConfirmedContext() || {};

    if (!dev || !date) {
      // Button should never be clickable without confirmed context,
      // but guard anyway.
      alert('Please wait for the SHAP analysis to finish loading first.');
      return;
    }

    // Snapshot into module state so closing/re-opening the SHAP panel
    // mid-simulation cannot change what the Run button posts.
    context.developer    = dev;
    context.date         = date;

    // Stamp the confirmed dev/date directly onto the Run button.
    // runSimulation() reads from there — never from context globals.
    const runBtn = el('btn-run-simulation');
    if (runBtn) {
      runBtn.dataset.dev  = dev;
      runBtn.dataset.date = date;
    }

    // Pre-fill current score from the SHAP panel's score ring
    const scoreText = document.getElementById('shap-content')
      ?.querySelector('[data-score]')?.dataset?.score;
    if (scoreText) context.currentScore = parseFloat(scoreText);

    open();
  }

  function closeSimulator() {
    el('simulator-modal').classList.add('hidden');
    if (animFrame) {
      cancelAnimationFrame(animFrame);
      animFrame = null;
    }
  }

  /* ─── Set context from heatmap click ─── */
  function setContext(developer, date, score) {
    context.developer    = developer;
    context.date         = date;
    context.currentScore = score;
  }

  /* ─── Update context bar text ─── */
  function updateContextBar() {
    const bar = el('sim-context');
    if (context.developer && context.date) {
      bar?.classList.remove('hidden');
      setText('sim-context-text',
        `${window.shortenEmail ? window.shortenEmail(context.developer) : context.developer} · ${context.date}`
      );
    } else {
      bar?.classList.add('hidden');
    }
  }

  /* ─── Update the "current score" display ─── */
  function updateCurrentScoreDisplay() {
    const scoreEl = el('sim-current-score');
    if (!scoreEl) return;

    if (context.currentScore !== null && context.currentScore !== undefined) {
      scoreEl.textContent = Math.round(context.currentScore);
      scoreEl.style.color = scoreColor(context.currentScore);
    } else {
      scoreEl.textContent = '—';
      scoreEl.style.color = '#6b7280';
    }

    // Reset projected
    setText('sim-projected-score', '—');
    el('sim-projected-score').style.color = '#6b7280';
    setText('sim-improvement-label', '—');
    el('sim-result-bar')?.classList.add('hidden');
  }

  /* ═══════════════════════════════════════════════════
     RUN SIMULATION
  ═══════════════════════════════════════════════════ */
  async function runSimulation() {
    const btn = el('btn-run-simulation');

    // Always read dev/date from the button's dataset — stamped by openFromShap()
    // using the confirmed context from ShapPanel. Never rely on stale globals.
    const developer = btn.dataset.dev;
    const date      = btn.dataset.date;

    if (!developer || !date) {
      alert('No developer / date context. Please reopen the simulator from a heatmap cell.');
      return;
    }

    btn.textContent = '⏳ Simulating...';
    btn.disabled = true;

    // Hide previous warning
    el('data-warning')?.classList.add('hidden');

    const changes = buildChangesObject();

    try {
      const body = { developer, date, changes };

      const res = await fetch(`${getApiBase()}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.ok) {
          displayResult(data.data);
        } else {
          displayMockResult(changes);
        }
      } else {
        displayMockResult(changes);
      }
    } catch {
      displayMockResult(changes);
    } finally {
      btn.innerHTML = 'Run Simulation';
      btn.disabled = false;
    }
  }

  /* ─── Build changes payload from sliders ─── */
  function buildChangesObject() {
    const changes = {};
    sliders.forEach(({ id, apiKey }) => {
      const slider = el(id);
      if (slider) changes[apiKey] = parseFloat(slider.value);
    });
    return changes;
  }

  /* ─── Display result from API ─── */
  function displayResult(data) {
    const current   = data.current_flow_score;
    const projected = Math.max(data.projected_flow_score, current); // never show lower
    const hours     = Math.max(0, data.estimated_hours_recovered_per_week ?? 0);
    const pts       = Math.max(0, data.improvement_points ?? (projected - current));
    const pct       = Math.max(0, data.improvement_percent ?? 0);
    const annual    = hours * 10 * 52;

    // Show data-availability warning if PR / CI data was missing during ingestion
    const avail = data.data_availability || {};
    const limitedData = !avail.pr_data && !avail.ci_data;
    el('data-warning')?.classList.toggle('hidden', !limitedData);

    // Update current score display
    const currentEl = el('sim-current-score');
    if (currentEl) {
      currentEl.textContent = Math.round(current);
      currentEl.style.color = scoreColor(current);
    }
    const currentLbl = el('sim-current-label');
    if (currentLbl) {
      currentLbl.textContent = scoreLabel(current);
      currentLbl.style.color = scoreColor(current);
    }

    // Animate projected score
    animateScoreChange(el('sim-projected-score'), current, projected);
    const projLbl = el('sim-projected-label');
    if (projLbl) {
      projLbl.textContent = `↑ +${pct.toFixed(1)}% improvement`;
      projLbl.style.color = '#10b981';
    }

    // Result bar metrics
    const deltaEl = el('sim-score-delta');
    if (deltaEl) deltaEl.textContent = pts > 0 ? `+${pts.toFixed(1)} pts` : 'No change';

    const pctEl = el('sim-improve-pct');
    if (pctEl) pctEl.textContent = pct > 0 ? `+${pct.toFixed(1)}%` : '—';

    const hoursEl = el('sim-hours-recovered');
    if (hoursEl) hoursEl.textContent = hours > 0 ? `${hours.toFixed(1)} h/week` : 'Needs PR/CI data';

    const annualEl = el('sim-annual-hours');
    if (annualEl) annualEl.textContent = annual > 0 ? `${Math.round(annual).toLocaleString()} hours/yr` : '—';

    el('sim-result-bar')?.classList.remove('hidden');
  }

  function scoreLabel(score) {
    if (score < 40) return '⚠ Disrupted';
    if (score < 65) return '~ Moderate';
    return '✓ Good';
  }

  /* ─── Fallback mock result ─── */
  function displayMockResult(changes) {
    const current = context.currentScore ?? 42;

    const prTimeGain  = Math.max(0, 24 - (changes.pr_blocking_time_hours || 24)) * 0.8;
    const prSizeGain  = Math.max(0, 600 - (changes.pr_size_lines || 600)) * 0.02;
    const buildGain   = Math.max(0, 20 - (changes.build_time_minutes || 20)) * 0.5;
    const focusGain   = Math.max(0, (changes.focus_block_minutes || 60) - 60) * 0.1;

    const totalGain  = Math.min(60, prTimeGain + prSizeGain + buildGain + focusGain);
    const projected  = Math.min(98, Math.max(current, current + totalGain + 5));
    const hours      = ((projected - current) / 100 * 8);

    // No data_availability key → the warning banner stays hidden.
    // The warning is only meaningful when the real API confirms PR/CI data
    // was absent during ingestion; mock results have no such context.
    displayResult({
      current_flow_score:                 current,
      projected_flow_score:               projected,
      improvement_points:                 projected - current,
      improvement_percent:                (projected - current) / Math.max(current, 1) * 100,
      estimated_hours_recovered_per_week: hours,
    });
  }

  /* ═══════════════════════════════════════════════════
     ANIMATED SCORE COUNTER
  ═══════════════════════════════════════════════════ */
  function animateScoreChange(targetEl, fromScore, toScore) {
    if (!targetEl) return;
    if (animFrame) cancelAnimationFrame(animFrame);

    const start     = performance.now();
    const duration  = 900; // ms
    const from      = Math.round(fromScore);
    const to        = Math.round(toScore);

    function easeOut(t) {
      return 1 - Math.pow(1 - t, 3);
    }

    function step(now) {
      const elapsed = now - start;
      const progress = Math.min(1, elapsed / duration);
      const eased = easeOut(progress);
      const current = Math.round(from + (to - from) * eased);

      targetEl.textContent = current;
      targetEl.style.color = scoreColor(current);

      if (progress < 1) {
        animFrame = requestAnimationFrame(step);
      } else {
        targetEl.textContent = to;
        targetEl.style.color = scoreColor(to);
        targetEl.classList.add('count-up');
        setTimeout(() => targetEl.classList.remove('count-up'), 400);
        animFrame = null;
      }
    }

    animFrame = requestAnimationFrame(step);
  }

  /* ═══════════════════════════════════════════════════
     RESET
  ═══════════════════════════════════════════════════ */
  function resetSimulator() {
    sliders.forEach(({ id, valId, format }) => {
      const slider = el(id);
      if (!slider) return;
      slider.value = slider.defaultValue;
      setText(valId, format(slider.defaultValue));
    });

    // Reset score displays
    const currentEl = el('sim-current-score');
    if (currentEl) {
      currentEl.textContent = context.currentScore !== null
        ? Math.round(context.currentScore)
        : '—';
      currentEl.style.color = context.currentScore ? scoreColor(context.currentScore) : '#6b7280';
    }

    setText('sim-projected-score', '—');
    el('sim-projected-score').style.color = '#6b7280';
    setText('sim-improvement-label', '—');
    el('sim-result-bar')?.classList.add('hidden');
  }

  /* ═══════════════════════════════════════════════════
     HELPERS
  ═══════════════════════════════════════════════════ */
  function scoreColor(score) {
    if (score >= 80) return '#10b981';
    if (score >= 60) return '#22c55e';
    if (score >= 40) return '#f97316';
    return '#ef4444';
  }

  function getApiBase() {
    return 'http://localhost:5000/api';
  }

  /* ─── Public API ─── */
  return { init, open, openFromShap, closeSimulator, setContext };

})();

/* ─── Auto-init ─── */
document.addEventListener('DOMContentLoaded', () => SimulatorModule.init());

window.SimulatorModule = SimulatorModule;