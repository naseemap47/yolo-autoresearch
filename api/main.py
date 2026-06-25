import os
import sys
import json
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.analyzer import analyze_yolo_dataset

# ── Configuration ─────────────────────────────────────────────────────────────
DB_PATH     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api_jobs.db"))
OUTPUT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolo_runs"))
DATA_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
UPLOADS_TMP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads_tmp"))

# In-memory process registry
running_processes: Dict[str, Any] = {}


# ── DB Helpers ────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(UPLOADS_TMP, exist_ok=True)
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at  TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS training_jobs (
            run_id          TEXT PRIMARY KEY,
            project_name    TEXT NOT NULL,
            experiment_name TEXT NOT NULL,
            dataset_path    TEXT NOT NULL,
            model_scale     TEXT NOT NULL,
            hyperparams     TEXT NOT NULL,
            status          TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            pid             INTEGER,
            output_dir      TEXT NOT NULL,
            metrics         TEXT
        )
    """)
    conn.commit()
    conn.close()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Mark stale 'running' jobs as 'failed' on restart
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT run_id, pid FROM training_jobs WHERE status = 'running'")
    for job in c.fetchall():
        pid = job["pid"]
        is_alive = False
        if pid:
            try:
                os.kill(pid, 0)
                is_alive = True
            except OSError:
                pass
        if not is_alive:
            c.execute("UPDATE training_jobs SET status='failed' WHERE run_id=?", (job["run_id"],))
    conn.commit()
    conn.close()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="YOLO Autoresearch Training API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Pydantic models ───────────────────────────────────────────────────────────
class ProjectCreate(BaseModel):
    name: str
    description: str = ""

class AnalyzeRequest(BaseModel):
    dataset_path: str

class TrainRequest(BaseModel):
    project_name: str
    experiment_name: str
    dataset_path: str
    model_scale: str          # full model id, e.g. "yolov8n", "yolo11s", "yolo26m"
    hyperparams: dict
    mlflow_uri: str = "http://localhost:5000"

class UploadInitRequest(BaseModel):
    dataset_name: str
    filename: str

class UploadFinalizeRequest(BaseModel):
    total_chunks: int


# ══════════════════════════════════════════════════════════════════════════════
# PROJECT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/projects")
def list_projects():
    conn = get_db()
    rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/projects", status_code=201)
def create_project(project: ProjectCreate):
    pid = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO projects (id, name, description, created_at) VALUES (?,?,?,?)",
            (pid, project.name, project.description, created_at)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(400, f"Project '{project.name}' already exists.")
    finally:
        conn.close()
    return {"id": pid, "name": project.name, "description": project.description, "created_at": created_at}


@app.delete("/api/projects/{project_name}")
def delete_project(project_name: str):
    conn = get_db()
    conn.execute("DELETE FROM projects WHERE name=?", (project_name,))
    conn.commit()
    conn.close()
    return {"message": f"Project '{project_name}' deleted."}


# ══════════════════════════════════════════════════════════════════════════════
# DATA UPLOAD ENDPOINTS  (chunked upload → ZIP extraction)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/data/upload/init")
def init_upload(request: UploadInitRequest):
    """Start a chunked upload session. Returns an upload_id."""
    upload_id = str(uuid.uuid4())
    tmp_dir = os.path.join(UPLOADS_TMP, upload_id)
    os.makedirs(tmp_dir, exist_ok=True)
    meta = {"dataset_name": request.dataset_name, "filename": request.filename}
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return {"upload_id": upload_id}


@app.post("/api/data/upload/chunk/{upload_id}")
async def upload_chunk(
    upload_id: str,
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...)
):
    """Receive one chunk and save it to the temp directory."""
    tmp_dir = os.path.join(UPLOADS_TMP, upload_id)
    if not os.path.isdir(tmp_dir):
        raise HTTPException(404, "Upload session not found.")
    content = await chunk.read()
    chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_index:06d}")
    with open(chunk_path, "wb") as f:
        f.write(content)
    return {"received": True, "chunk_index": chunk_index, "size": len(content)}


@app.post("/api/data/upload/finalize/{upload_id}")
def finalize_upload(upload_id: str, request: UploadFinalizeRequest):
    """Assemble all chunks → extract ZIP → return server_path."""
    tmp_dir = os.path.join(UPLOADS_TMP, upload_id)
    if not os.path.isdir(tmp_dir):
        raise HTTPException(404, "Upload session not found.")

    with open(os.path.join(tmp_dir, "meta.json")) as f:
        meta = json.load(f)
    dataset_name = meta["dataset_name"]
    filename = meta["filename"]

    # Assemble
    assembled = os.path.join(tmp_dir, filename)
    with open(assembled, "wb") as out:
        for i in range(request.total_chunks):
            chunk_path = os.path.join(tmp_dir, f"chunk_{i:06d}")
            if not os.path.exists(chunk_path):
                raise HTTPException(400, f"Missing chunk {i}.")
            with open(chunk_path, "rb") as cf:
                out.write(cf.read())

    # Extract
    dataset_dir = os.path.join(DATA_ROOT, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(assembled, "r") as zf:
            zf.extractall(dataset_dir)
    except zipfile.BadZipFile:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, "Uploaded file is not a valid ZIP archive.")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Locate yaml
    yaml_path = None
    for root, _dirs, files in os.walk(dataset_dir):
        for fname in files:
            if fname.endswith((".yaml", ".yml")):
                yaml_path = os.path.join(root, fname)
                break
        if yaml_path:
            break

    return {"dataset_name": dataset_name, "server_path": dataset_dir, "yaml_path": yaml_path}


@app.get("/api/data/list")
def list_datasets():
    """List all datasets that have been uploaded to DATA_ROOT."""
    datasets = []
    if not os.path.isdir(DATA_ROOT):
        return datasets
    for name in sorted(os.listdir(DATA_ROOT)):
        dataset_dir = os.path.join(DATA_ROOT, name)
        if not os.path.isdir(dataset_dir):
            continue
        yaml_path = None
        total_size = 0
        img_count = 0
        for root, _dirs, files in os.walk(dataset_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                total_size += os.path.getsize(fpath)
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    img_count += 1
                if yaml_path is None and fname.endswith((".yaml", ".yml")):
                    yaml_path = fpath
        datasets.append({
            "dataset_name": name,
            "server_path": dataset_dir,
            "yaml_path": yaml_path,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "image_count": img_count,
        })
    return datasets


@app.delete("/api/data/{dataset_name}")
def delete_dataset(dataset_name: str):
    dataset_dir = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(404, "Dataset not found.")
    shutil.rmtree(dataset_dir)
    return {"message": f"Dataset '{dataset_name}' deleted."}


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/analyze-dataset")
def analyze_dataset(request: AnalyzeRequest):
    if not os.path.exists(request.dataset_path):
        raise HTTPException(400, f"Dataset path does not exist: {request.dataset_path}")
    try:
        return analyze_yolo_dataset(request.dataset_path)
    except Exception as e:
        raise HTTPException(500, f"Error analyzing dataset: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/jobs")
def list_jobs():
    conn = get_db()
    rows = conn.execute("SELECT * FROM training_jobs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/train")
def start_training(request: TrainRequest):
    run_id = f"{request.project_name}_{request.experiment_name}".replace(" ", "_")
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT status FROM training_jobs WHERE run_id=?", (run_id,))
    existing = c.fetchone()
    if existing and existing["status"] == "running":
        conn.close()
        raise HTTPException(400, "This experiment is already running.")

    run_dir = os.path.join(OUTPUT_DIR, request.project_name, request.experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "training.log")
    log_file = open(log_path, "w")

    worker = os.path.join(os.path.dirname(__file__), "train_worker.py")
    cmd = [
        sys.executable, worker,
        "--project-name",    request.project_name,
        "--experiment-name", request.experiment_name,
        "--dataset-path",    request.dataset_path,
        "--model-scale",     request.model_scale,
        "--hyperparams",     json.dumps(request.hyperparams),
        "--output-dir",      os.path.join(OUTPUT_DIR, request.project_name),
        "--mlflow-uri",      request.mlflow_uri,
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, bufsize=1)
        running_processes[run_id] = proc
        c.execute(
            """INSERT OR REPLACE INTO training_jobs
               (run_id,project_name,experiment_name,dataset_path,model_scale,hyperparams,status,created_at,pid,output_dir)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (run_id, request.project_name, request.experiment_name, request.dataset_path,
             request.model_scale, json.dumps(request.hyperparams), "running",
             datetime.now().isoformat(), proc.pid, os.path.join(OUTPUT_DIR, request.project_name))
        )
        conn.commit()
    except Exception as e:
        log_file.close()
        conn.close()
        raise HTTPException(500, f"Failed to start training: {e}")
    finally:
        conn.close()

    return {"run_id": run_id, "status": "running", "pid": proc.pid, "log_file": log_path}


@app.get("/api/train/status/{run_id}")
def get_status(run_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT project_name,experiment_name,pid,status,output_dir,hyperparams FROM training_jobs WHERE run_id=?",
        (run_id,)
    )
    job = c.fetchone()
    conn.close()
    if not job:
        raise HTTPException(404, "Training job not found.")

    status      = job["status"]
    output_dir  = job["output_dir"]
    exp_name    = job["experiment_name"]
    pid         = job["pid"]
    total_epochs = json.loads(job["hyperparams"]).get("epochs", 10)

    run_dir     = os.path.join(output_dir, exp_name)
    status_file = os.path.join(run_dir, "status.json")
    csv_path    = os.path.join(run_dir, "results.csv")

    if status == "running":
        proc = running_processes.get(run_id)
        alive = (proc.poll() is None) if proc else False
        if not alive and pid:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                pass
        if not alive:
            if os.path.exists(status_file):
                try:
                    with open(status_file) as sf:
                        sd = json.load(sf)
                    status = sd.get("status", "completed")
                    conn = get_db()
                    conn.execute("UPDATE training_jobs SET status=?,metrics=? WHERE run_id=?",
                                 (status, json.dumps(sd.get("metrics", {})), run_id))
                    conn.commit()
                    conn.close()
                except Exception:
                    status = "failed"
            else:
                status = "failed"
                conn = get_db()
                conn.execute("UPDATE training_jobs SET status='failed' WHERE run_id=?", (run_id,))
                conn.commit()
                conn.close()

    current_epoch = 0
    progress_metrics: dict = {}
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = [col.strip() for col in df.columns]
            current_epoch = len(df)
            if current_epoch > 0:
                last = df.iloc[-1]
                progress_metrics = {
                    "epoch":          current_epoch,
                    "train_box_loss": float(last.get("train/box_loss",          0.0)),
                    "val_box_loss":   float(last.get("val/box_loss",            0.0)),
                    "map50":          float(last.get("metrics/mAP50(B)",        0.0)),
                    "map50_95":       float(last.get("metrics/mAP50-95(B)",     0.0)),
                }
        except Exception:
            pass

    log_tail: List[str] = []
    log_path = os.path.join(run_dir, "training.log")
    if os.path.exists(log_path):
        try:
            with open(log_path) as lf:
                log_tail = lf.readlines()[-30:]
        except Exception:
            pass

    return {
        "run_id": run_id,
        "status": status,
        "progress": {
            "current_epoch": current_epoch,
            "total_epochs":  total_epochs,
            "pct": round((current_epoch / total_epochs) * 100, 2) if total_epochs > 0 else 0.0,
        },
        "metrics": progress_metrics,
        "logs": "".join(log_tail),
    }


@app.get("/api/train/results/{run_id}")
def get_results(run_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT project_name,experiment_name,status,output_dir,metrics FROM training_jobs WHERE run_id=?", (run_id,))
    job = c.fetchone()
    conn.close()
    if not job:
        raise HTTPException(404, "Training job not found.")
    metrics: dict = {}
    if job["metrics"]:
        metrics = json.loads(job["metrics"])
    elif os.path.exists(os.path.join(job["output_dir"], job["experiment_name"], "status.json")):
        try:
            with open(os.path.join(job["output_dir"], job["experiment_name"], "status.json")) as sf:
                metrics = json.load(sf).get("metrics", {})
        except Exception:
            pass
    return {"run_id": run_id, "project_name": job["project_name"],
            "experiment_name": job["experiment_name"], "status": job["status"], "metrics": metrics}


@app.get("/api/train/download/{run_id}")
def download_run_artifacts(run_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT project_name,experiment_name,output_dir FROM training_jobs WHERE run_id=?", (run_id,))
    job = c.fetchone()
    conn.close()
    if not job:
        raise HTTPException(404, "Training job not found.")
    run_dir = os.path.join(job["output_dir"], job["experiment_name"])
    if not os.path.isdir(run_dir):
        raise HTTPException(404, "Run output directory not found.")
    tmp = tempfile.mkdtemp()
    try:
        archive = shutil.make_archive(os.path.join(tmp, f"{run_id}_artifacts"), "zip", run_dir)
        return FileResponse(archive, media_type="application/zip", filename=f"{run_id}_artifacts.zip")
    except Exception as e:
        raise HTTPException(500, f"Failed to package artifacts: {e}")


@app.post("/api/train/stop/{run_id}")
def stop_training(run_id: str):
    proc = running_processes.get(run_id)
    conn = get_db()
    c = conn.cursor()
    if not proc:
        c.execute("SELECT pid,status FROM training_jobs WHERE run_id=?", (run_id,))
        job = c.fetchone()
        if not job:
            conn.close()
            raise HTTPException(404, "Job not found.")
        if job["status"] != "running":
            conn.close()
            return {"message": "Job is not running", "status": job["status"]}
    terminated = False
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=5)
            terminated = True
        except Exception:
            try:
                proc.kill()
                terminated = True
            except Exception:
                pass
    c.execute("UPDATE training_jobs SET status='stopped' WHERE run_id=?", (run_id,))
    conn.commit()
    conn.close()
    running_processes.pop(run_id, None)
    return {"message": "Training stopped.", "terminated": terminated}