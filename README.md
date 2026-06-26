# YOLO AutoResearch — Agentic Framework

> Automated YOLO model selection, hyperparameter tuning, and iterative training — powered by LangGraph + Ollama.

---

## Architecture

```
┌─────────────────────────────────┐        ┌──────────────────────────────────┐
│        LLM Node (local)         │        │      GPU Training Node           │
│                                 │        │                                  │
│  ┌──────────────────────────┐   │  HTTP  │  ┌───────────────────────────┐   │
│  │  Static HTML/JS Frontend │◄──┼────────┼─►│  FastAPI Training API     │   │
│  │  (frontend/index.html)   │   │        │  │  (api/main.py)            │   │
│  └──────────────────────────┘   │        │  └─────────────┬─────────────┘   │
│  ┌──────────────────────────┐   │        │                │ subprocess      │
│  │  FastAPI Orchestrator    │   │        │  ┌─────────────▼─────────────┐   │
│  │  (orchestrator.py)       │   │        │  │  YOLO Train Worker        │   │
│  └──────────────────────────┘   │        │  │  (api/train_worker.py)    │   │
│  ┌──────────────────────────┐   │        │  └─────────────┬─────────────┘   │
│  │  Ollama LLM              │   │        │                │ logs/metrics    │
│  │  (local, port 11434)     │   │        │  ┌─────────────▼─────────────┐   │
│  └──────────────────────────┘   │        │  │  MLflow Tracking Server   │   │
└─────────────────────────────────┘        │  │  (port 5000)              │   │
                                           │  └───────────────────────────┘   │
                                           └──────────────────────────────────┘
```

## Research Loop

```
Dataset Upload
    │
    ▼
[1] Analyze Data        ← bbox distribution (small/medium/large), class count, image count
    │
    ▼
[2] LLM Select Model    ← Ollama picks YOLO family (yolov5→yolo26) + scale (n/s/m/l/x) + hyperparams
    │
    ▼
[3] YOLO Train          ← FastAPI spawns subprocess on GPU node, streams logs
    │
    ▼
[4] Evaluate            ← mAP50, precision, recall, box losses
    │
    ├─ mAP50 ≥ target  → ✅ DONE
    ├─ cycles = N       → ✅ DONE
    │
    ▼
[5] LLM Adjust Model    ← Diagnoses overfitting/underfitting, selects new model + hyperparams
    │
    └──────────────────► [3] Train again
```

---

## Project Structure

```
yolo-autoresearch/
├── api/
│   ├── main.py          # FastAPI training server (runs on GPU node)
│   ├── train_worker.py  # YOLO training subprocess worker
│   └── analyzer.py      # Dataset analysis (bbox stats, class dist)
├── orchestrator.py    # FastAPI orchestrator server (runs LangGraph + serves frontend)
├── frontend/          # HTML/JS UI
│   ├── index.html     # Main HTML layout
│   ├── style.css      # Custom UI styles
│   └── app.js         # Frontend logic
├── datasets/          # Uploaded datasets (on GPU node)
├── yolo_runs/           # Training outputs (weights, logs, charts)
├── mlruns/              # MLflow artifacts
├── docker-compose.yml
└── main.py              # CLI runner
```

---

## Quick Start (Single Machine)

### 1. Install dependencies

```bash
pip install uv
uv sync   # installs from pyproject.toml
```

Or with pip:
```bash
pip install fastapi uvicorn langgraph langchain-core \
            ultralytics mlflow requests pandas pyyaml matplotlib python-multipart
```

### 2. Install & start Ollama

```bash
# Install Ollama: https://ollama.com
ollama pull qwen2.5:3b
ollama serve   # runs on http://localhost:11434
```

### 3. Start all services

```bash
python main.py start-all
```

This starts:
- **MLflow** → http://localhost:5000
- **FastAPI Training API** → http://localhost:8000
- **HTML/JS UI** → http://localhost:8501

---

## Two-Machine Setup (Recommended)

### GPU Node — Training Server

```bash
# Clone repo on the GPU machine
git clone <repo>
cd yolo-autoresearch

# Start MLflow + FastAPI
python main.py mlflow &
python main.py api
```

Or with Docker:
```bash
docker compose up training-api mlflow
```

### LLM Node — Local Machine

Set the URLs in the UI sidebar to point to your GPU machine:
```
http://<GPU_IP>:8000   ← Training API
http://<GPU_IP>:5000   ← MLflow
```

Then start locally:
```bash
python main.py frontend
```

---

## Workflow: Using the UI

### 1. Create a Project
- Sidebar → **Project** → "Create New Project"
- Enter name and description → **Create Project**

### 2. Upload Dataset
- Sidebar → **Dataset** → "Upload Dataset (ZIP)"
- ZIP must contain YOLO format: `train/images/`, `train/labels/`, `val/images/`, `val/labels/`, `data.yaml`
- Uses **chunked upload** (2MB chunks) — works over slow networks

### 3. Start Autoresearch
- Select your project and dataset
- Set **Target mAP50** and **Max Cycles**
- Click **▶ Start Autoresearch**

### 4. Monitor
- **🔬 Research Loop** tab — stepper, live logs, epoch progress, cycle history with LLM reasoning
- **📈 MLflow** tab — embedded MLflow UI for metric comparison across all runs
- **📦 Data Management** tab — list, inspect, delete datasets on the training server
- **🗃️ History & Downloads** tab — all runs table, download `best.pt` + `results.csv` as ZIP

---

## Supported YOLO Models

The LLM agent selects from any combination of:

| Family | Scales | Notes |
|--------|--------|-------|
| `yolov5` | n, s, m, l, x | Stable, widely tested |
| `yolov6` | n, s, m, l | Meituan's fork |
| `yolov8` | n, s, m, l, x | **Recommended default** |
| `yolo11` | n, s, m, l, x | Ultralytics YOLO11 |
| `yolo12` | n, s, m, l, x | Attention-based |
| `yolo26` | n, s, m, l, x | Cutting-edge |

Custom models (e.g., `yolo26n.pt`) can be placed in the project root and referenced by name.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL`  | `qwen2.5:3b`            | Ollama model to use |

---

## API Reference (Training Server)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/projects` | List projects |
| POST | `/api/projects` | Create project |
| GET  | `/api/data/list` | List uploaded datasets |
| POST | `/api/data/upload/init` | Start chunked upload |
| POST | `/api/data/upload/chunk/{id}` | Upload chunk |
| POST | `/api/data/upload/finalize/{id}` | Finalize & extract |
| DELETE | `/api/data/{name}` | Delete dataset |
| POST | `/api/analyze-dataset` | Analyze YOLO dataset |
| POST | `/api/train` | Start training job |
| GET  | `/api/train/status/{run_id}` | Get job status + logs |
| GET  | `/api/train/results/{run_id}` | Get final metrics |
| GET  | `/api/train/download/{run_id}` | Download artifacts ZIP |
| POST | `/api/train/stop/{run_id}` | Stop training |
| GET  | `/api/jobs` | List all jobs |

Interactive docs: http://localhost:8000/docs