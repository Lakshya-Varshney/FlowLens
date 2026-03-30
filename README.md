# FlowLens

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Deploy on Railway](https://img.shields.io/badge/Deploy-Railway-blueviolet.svg)](https://flowlens.up.railway.app)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red.svg)](https://github.com/)
[![DX-Ray Hackathon 2026](https://img.shields.io/badge/DX--Ray%20Hackathon%202026-Track%20F-gold.svg)](#)

**AI-powered developer flow intelligence. Expose hidden productivity disruptions — locally, privately, in minutes.**

FlowLens ingests your Git history, runs ML anomaly detection to surface invisible developer friction, explains every anomaly with SHAP values, and prescribes specific workflow fixes. No telemetry. No SaaS. Everything runs on your machine.

> 🏆 **Winner — Track F (Developer Flow Scan)** · DX-Ray Hackathon 2026 (March 27–30)  
> 🔗 **Cross-track integration** with Track B (Test Health X-Ray)

---

## The Problem

Developer productivity loss is mostly invisible. Interrupted focus blocks, oversized PRs, late-night commit spikes, and slow CI pipelines don't show up in sprint metrics — they show up as burnout, rework, and missed deadlines. Teams have no instrument to detect them, let alone fix them.

FlowLens puts hard numbers on invisible DX friction.

---

## Features

**📡 Smart Git Ingestion**
- Blobless, bare clone (`--filter=blob:none`) — skips vendor/staging directories entirely
- Kubernetes-scale repos clone in ~2 minutes instead of 30
- Repo caching: first run clones, subsequent runs do a fast `git fetch` (~10 seconds)
- Supports public GitHub URLs, private repos (via `GITHUB_TOKEN`), and local paths

**🧠 ML Anomaly Detection**
- Isolation Forest trained on 14 engineered features per developer per day
- SHAP explainability: every anomaly score is decomposed into top contributing factors with actual vs. baseline values
- No black boxes — you see exactly *why* a day was flagged

**📊 Flow Score Dashboard**
- GitHub-style heatmap calendar, colour-coded by flow score (Low / Mid / High)
- 7-day rolling average trend chart
- Per-developer individual scores
- Full anomaly log

**🔬 Flow X-Ray Diagnosis Panel**
- Click any heatmap cell to open a side panel
- Score circle + severity label + recoverable hours/week
- SHAP bar chart with factor-level explanations
- Plain-text diagnosis, numbered rule-based actions, and an AI Recommendation section (WHAT HAPPENED / PRIORITISED ACTIONS / HOURS RECOVERED)

**⚡ Before/After Simulator**
- Interactive sliders: PR Review Time Target, Max PR Size, Build Time Target, Focus Block Length
- Animated count-up from current score → projected score
- Shows hours recovered per week and annualised impact for a 10-developer team

**🔒 Privacy-First**
- All processing is local by default
- The only optional external call is to HuggingFace Inference API (for AI recommendations) — fully opt-in via `HF_TOKEN`
- No data ever leaves your machine unless you explicitly configure it

---

## Quick Start

### Prerequisites

- Python 3.11+
- Git

### 1. Clone and install

```bash
git clone https://github.com/your-org/flowlens.git
cd flowlens
pip install -r requirements.txt
```

### 2. Run the instant demo

```bash
python run.py --demo
```

Opens the dashboard pre-loaded with synthetic data. No GitHub token, no network access required.

### 3. Scan a real repository

```bash
python run.py --repo https://github.com/org/repo
```

Then open [http://localhost:5000](http://localhost:5000), paste your repo URL, and click **Scan**.

### 4. Optional: Enable AI recommendations

```bash
export HF_TOKEN=hf_...          # huggingface.co/settings/tokens
                                # Requires "Make calls to Inference Providers" permission
python run.py --repo https://github.com/org/repo
```

---

## CLI Reference

```bash
# Instant demo with synthetic data
python run.py --demo

# Scan a GitHub repository (last 90 days by default)
python run.py --repo https://github.com/org/repo

# Scan from a specific date
python run.py --repo https://github.com/org/repo --since 2025-12-01

# Scan with a custom commit depth
python run.py --repo https://github.com/org/repo --depth 1000

# Scan a local repository (all 14 features active, no token needed)
python run.py --repo /path/to/local/repo
```

---

## Docker

```bash
docker build -t flowlens .
docker run -p 5000:5000 flowlens
```

To pass environment variables:

```bash
docker run -p 5000:5000 \
  -e HF_TOKEN=hf_... \
  -e GITHUB_TOKEN=ghp_... \
  flowlens
```

---

## Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://flowlens.up.railway.app)

**Live demo:** [flowlens.up.railway.app](https://flowlens.up.railway.app)

The app is production-ready and deployable in a single click. Set `HF_TOKEN` and `GITHUB_TOKEN` as Railway environment variables for full feature coverage.

---

## Environment Variables

All variables are optional. FlowLens is fully functional without any of them.

| Variable | Purpose | Where to get it |
|---|---|---|
| `HF_TOKEN` | Enables AI recommendations via Qwen2.5-72B | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — enable "Make calls to Inference Providers" |
| `GITHUB_TOKEN` | Unlocks PR review time and PR size features (4 additional ML features) | GitHub → Settings → Developer settings → Personal access tokens |
| `SLACK_WEBHOOK_URL` | Sends a daily flow digest to a Slack channel | Slack app configuration → Incoming Webhooks |

> **Without `GITHUB_TOKEN`:** 10 of 14 features are active. Timestamp-based features (session duration, gap variance, late-night ratio, etc.) still detect anomalies effectively.  
> **Without `HF_TOKEN`:** Rule-based recommendations are shown in place of AI-generated ones.

---

## ML Features

FlowLens engineers 14 features per developer per calendar day from raw Git metadata:

| Feature | Source | Requires |
|---|---|---|
| `session_duration_minutes` | Commit timestamps | Git only |
| `inter_commit_gap_mean` | Commit timestamps | Git only |
| `inter_commit_gap_variance` | Commit timestamps | Git only |
| `late_night_ratio` | Commit hour distribution | Git only |
| `weekend_ratio` | Commit day-of-week | Git only |
| `merge_commit_ratio` | Commit parents | Git only |
| `commit_message_length_mean` | Commit messages | Git only |
| `daily_commit_count` | Commit count | Git only |
| `files_changed_per_commit` | Diff stats | Git only |
| `lines_changed_per_commit` | Diff stats | Git only |
| `pr_review_time_hours` | PR events | `GITHUB_TOKEN` |
| `pr_size_lines` | PR diff | `GITHUB_TOKEN` |
| `ci_build_time_minutes` | CI logs | `GITHUB_TOKEN` + CI |
| `ci_failure_ratio` | CI outcomes | `GITHUB_TOKEN` + CI |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ingest` | Start background ingestion; returns `job_id` |
| `GET` | `/api/ingestion-status` | Poll pipeline progress (`step`, `percent`, `message`) |
| `GET` | `/api/status` | System health check |
| `GET` | `/api/flow-scores` | Heatmap data for all developers |
| `GET` | `/api/shap/<date>/<developer>` | SHAP explanation for a specific developer-day |
| `GET` | `/api/recommend/<date>/<developer>` | Rule-based + LLM recommendations |
| `POST` | `/api/simulate` | Before/After workflow simulator |
| `GET` | `/api/trend` | 7-day rolling average trend |
| `GET` | `/api/developers` | List of detected developers |

---

## Configuration

All tunable parameters live in `config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `model.contamination` | `0.1` | Expected fraction of anomalous days (IsolationForest) |
| `model.n_estimators` | `200` | Number of trees in the forest |
| `features.session_gap_minutes` | `120` | Inactivity gap that ends a coding session |
| `ingest.default_depth` | `500` | Default commit history depth |
| `ingest.cache_dir` | `./data/repos` | Local path for cloned repo cache |
| `server.port` | `5000` | Flask server port |
| `server.debug` | `false` | Enable Flask debug mode |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FlowLens Pipeline                        │
└─────────────────────────────────────────────────────────────────┘

  GitHub URL / Local Path
        │
        ▼
  ┌─────────────┐     bare + blobless clone     ┌──────────────┐
  │  ingest.py  │ ◄────────────────────────────► │  Git Cache   │
  │             │     fast fetch on re-runs      │  ./data/repos│
  └──────┬──────┘                                └──────────────┘
         │  raw commits + metadata
         ▼
  ┌─────────────┐
  │ features.py │  14-feature engineering per developer per day
  └──────┬──────┘
         │  feature matrix  (pandas DataFrame)
         ▼
  ┌─────────────┐
  │   model.py  │  IsolationForest  →  anomaly scores
  │             │  SHAP explainer   →  factor contributions
  └──────┬──────┘
         │  scored + explained developer-days
         ▼
  ┌──────────────┐
  │  insights.py │  Rule-based recommendations
  │              │  + optional LLM (Qwen2.5-72B via HF Inference)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │    db.py     │  SQLite persistence
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐          ┌───────────────────────────────┐
  │    api.py    │ ◄───────► │  Frontend (HTML/CSS/JS)       │
  │  Flask REST  │          │  · Heatmap calendar           │
  └──────────────┘          │  · SHAP diagnosis panel       │
                            │  · Before/After simulator     │
                            │  · Trend chart (Chart.js)     │
                            └───────────────────────────────┘
```

---

## Project Structure

```
flowlens/
├── run.py                  # CLI entry point
├── config.yaml             # All tunable parameters
├── requirements.txt
├── Dockerfile
├── flowlens/
│   ├── api.py              # Flask REST API
│   ├── db.py               # SQLite persistence layer
│   ├── ingest.py           # Git ingestion (bare + blobless clone, repo caching)
│   ├── features.py         # 14-feature developer-day engineering
│   ├── model.py            # IsolationForest + SHAP
│   ├── insights.py         # Rule-based + LLM recommendations
│   ├── simulator.py        # Before/After workflow simulator
│   └── demo.py             # Synthetic demo dataset generator
└── frontend/
    ├── home.html           # Landing page with repo input + live progress bar
    └── index.html          # Main dashboard
```

---

## Hackathon Judging Criteria

| Criterion | Weight | How FlowLens addresses it |
|---|---|---|
| Problem Diagnosis | 25% | Quantifies invisible DX friction with hard numbers — anomaly scores, hours lost, feature-level attribution |
| Solution Impact | 25% | SHAP-explained recommendations + interactive Before/After simulator with annualised team projections |
| Technical Execution | 20% | End-to-end ML pipeline: IsolationForest → SHAP → LLM (Qwen2.5-72B) with 14 engineered features |
| User Experience | 15% | Single-command setup, local-first, privacy-preserving, live 8-stage progress bar |
| Presentation & Demo | 15% | `--demo` mode, live Railway deployment, full dashboard on first launch |
| **Bonus** | | |
| Real Data Demo | +5% | Works on any public GitHub repo out of the box |
| Before/After Metrics | +5% | Animated simulator with hours recovered and annual team impact |
| Open Source | +3% | MIT licensed |
| Cross-Track B Integration | +3% | CI failure ratio and test health metrics feed directly into the ML feature set |

---

## Contributing

Contributions are welcome. To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests where applicable
4. Run the demo to verify nothing is broken: `python run.py --demo`
5. Open a pull request with a clear description of what changed and why

For larger changes, open an issue first to discuss the approach.

### Development setup

```bash
git clone https://github.com/your-org/flowlens.git
cd flowlens
pip install -r requirements.txt
python run.py --demo        # verify baseline works
```

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.

---

*Built in 72 hours for the DX-Ray Hackathon 2026. All analysis runs locally. Your code never leaves your machine.*
