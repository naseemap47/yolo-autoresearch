import os
import io
import json
import math
import time
import threading
import requests
import streamlit as st
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLO AutoResearch",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB per chunk

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main-title{background:linear-gradient(135deg,#6C63FF,#FF6584);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;font-weight:700;font-size:2.4rem;margin-bottom:.1rem;}
.subtitle{color:#888;font-size:1rem;margin-bottom:1.5rem;}
.card{background:#1a1a2e;padding:1.2rem;border-radius:12px;border:1px solid #2d2d4a;
  box-shadow:0 4px 16px rgba(0,0,0,.3);margin-bottom:.8rem;}
.metric-val{font-size:1.7rem;font-weight:700;color:#6C63FF;}
.metric-lbl{font-size:.78rem;color:#aaa;text-transform:uppercase;letter-spacing:.5px;}
.log-box{max-height:300px;overflow-y:auto;font-family:monospace;background:#0a0a14;
  color:#00e676;padding:12px;border-radius:8px;border:1px solid #1a1a2e;font-size:.8rem;line-height:1.4;}
.step-row{display:flex;align-items:center;gap:0;margin-bottom:1.5rem;}
.step{display:flex;flex-direction:column;align-items:center;flex:1;position:relative;}
.step:not(:last-child)::after{content:'';position:absolute;top:18px;left:60%;
  width:80%;height:2px;background:#2d2d4a;z-index:1;}
.step.done:not(:last-child)::after{background:#4CAF50;}
.step.active:not(:last-child)::after{background:#6C63FF55;}
.sbadge{width:36px;height:36px;border-radius:50%;background:#1a1a2e;border:2px solid #2d2d4a;
  color:#aaa;display:flex;align-items:center;justify-content:center;font-weight:700;
  z-index:2;margin-bottom:.3rem;font-size:.85rem;}
.step.active .sbadge{background:#6C63FF;border-color:#6C63FF;color:#fff;
  box-shadow:0 0 14px rgba(108,99,255,.5);}
.step.done .sbadge{background:#4CAF50;border-color:#4CAF50;color:#fff;}
.slabel{font-size:.75rem;color:#888;font-weight:500;}
.step.active .slabel{color:#6C63FF;font-weight:600;}
.step.done .slabel{color:#4CAF50;}
.badge-running{background:#FF6584;color:#fff;padding:2px 10px;border-radius:20px;font-size:.75rem;}
.badge-done{background:#4CAF50;color:#fff;padding:2px 10px;border-radius:20px;font-size:.75rem;}
.badge-fail{background:#f44336;color:#fff;padding:2px 10px;border-radius:20px;font-size:.75rem;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "agent_running": False,
    "agent_state": {},
    "selected_project": None,
    "selected_dataset_path": None,
    "selected_dataset_yaml": None,
    "upload_done": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── API helpers ───────────────────────────────────────────────────────────────
def api_get(api_url, path, timeout=10):
    try:
        r = requests.get(f"{api_url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_post(api_url, path, payload=None, timeout=10):
    try:
        r = requests.post(f"{api_url}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_delete(api_url, path, timeout=10):
    try:
        r = requests.delete(f"{api_url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def chunked_upload(api_url: str, file_bytes: bytes, dataset_name: str, filename: str):
    """Upload file bytes in 2MB chunks to the training API."""
    # Init
    init_r = requests.post(f"{api_url}/api/data/upload/init",
                           json={"dataset_name": dataset_name, "filename": filename}, timeout=15)
    init_r.raise_for_status()
    upload_id = init_r.json()["upload_id"]

    total = len(file_bytes)
    n_chunks = math.ceil(total / CHUNK_SIZE)

    prog = st.progress(0.0, text="Uploading…")
    for i in range(n_chunks):
        chunk = file_bytes[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        buf = io.BytesIO(chunk)
        r = requests.post(
            f"{api_url}/api/data/upload/chunk/{upload_id}",
            data={"chunk_index": i},
            files={"chunk": (f"chunk_{i}", buf, "application/octet-stream")},
            timeout=60,
        )
        r.raise_for_status()
        prog.progress((i + 1) / n_chunks, text=f"Uploading chunk {i+1}/{n_chunks}…")

    # Finalize
    fin_r = requests.post(
        f"{api_url}/api/data/upload/finalize/{upload_id}",
        json={"total_chunks": n_chunks},
        timeout=60,
    )
    fin_r.raise_for_status()
    prog.empty()
    return fin_r.json()


# ── Background workflow runner ────────────────────────────────────────────────
def run_workflow_bg(inputs: dict, ollama_url: str):
    from core.graph import workflow_app
    import core.graph as cg
    cg.OLLAMA_DEFAULT_URL = ollama_url
    st.session_state.agent_state = inputs
    try:
        for output in workflow_app.stream(inputs):
            for _node, update in output.items():
                st.session_state.agent_state.update(update)
                if update.get("error"):
                    break
        if not st.session_state.agent_state.get("error"):
            st.session_state.agent_state["status"] = "Autoresearch completed."
    except Exception as e:
        st.session_state.agent_state["error"] = str(e)
        st.session_state.agent_state["status"] = f"Workflow error: {e}"
    finally:
        st.session_state.agent_running = False


# ── Stepper renderer ──────────────────────────────────────────────────────────
def render_stepper(state: dict):
    status = state.get("status", "")
    has_stats  = bool(state.get("data_stats"))
    has_model  = bool(state.get("current_model"))
    history    = state.get("history", [])
    cycle      = state.get("current_cycle", 0)

    s1 = "done" if has_stats else ("active" if "analyz" in status.lower() else "")
    s2 = "done" if has_model else ("active" if "select" in status.lower() else "")
    s3 = "done" if len(history) >= cycle > 0 else ("active" if "train" in status.lower() else "")
    s4 = "done" if len(history) >= cycle > 0 else ("active" if "evaluat" in status.lower() else "")
    s5 = "active" if "adjust" in status.lower() or "tuning" in status.lower() else (
         "done" if cycle > 1 and len(history) > 0 else "")

    steps = [("1", "Data Analysis", s1), ("2", "LLM Selects", s2),
             ("3", "YOLO Train", s3), ("4", "Evaluate", s4), ("5", "LLM Adjust", s5)]

    html = '<div class="step-row">'
    for num, label, cls in steps:
        html += f'<div class="step {cls}"><div class="sbadge">{num}</div><div class="slabel">{label}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## ⚙️ System Settings")
api_url    = st.sidebar.text_input("Training API URL",  "http://localhost:8000")
mlflow_uri = st.sidebar.text_input("MLflow URL",        "http://localhost:5000")
ollama_url = st.sidebar.text_input("Ollama URL",        "http://localhost:11434")

st.sidebar.divider()

# ── Project Management ────────────────────────────────────────────────────────
st.sidebar.markdown("## 📁 Project")
projects_data, _ = api_get(api_url, "/api/projects")
project_names = [p["name"] for p in (projects_data or [])]

with st.sidebar.expander("➕ Create New Project", expanded=not bool(project_names)):
    new_proj_name = st.text_input("Name", key="new_proj_name")
    new_proj_desc = st.text_area("Description", key="new_proj_desc", height=70)
    if st.button("Create Project", key="btn_create_proj"):
        if new_proj_name.strip():
            res, err = api_post(api_url, "/api/projects",
                                {"name": new_proj_name.strip(), "description": new_proj_desc.strip()})
            if err:
                st.error(f"Error: {err}")
            else:
                st.success(f"✅ Project '{new_proj_name}' created!")
                st.rerun()
        else:
            st.warning("Enter a project name.")

if project_names:
    selected_project = st.sidebar.selectbox("Select Project", project_names, key="proj_select")
    st.session_state.selected_project = selected_project
    proj_info = next((p for p in (projects_data or []) if p["name"] == selected_project), {})
    if proj_info.get("description"):
        st.sidebar.caption(f"📝 {proj_info['description']}")
else:
    st.sidebar.info("Create a project first.")
    st.session_state.selected_project = None

st.sidebar.divider()

# ── Data Management ───────────────────────────────────────────────────────────
st.sidebar.markdown("## 📦 Dataset")
datasets_data, _ = api_get(api_url, "/api/data/list")
dataset_names = [d["dataset_name"] for d in (datasets_data or [])]

with st.sidebar.expander("⬆️ Upload Dataset (ZIP)", expanded=not bool(dataset_names)):
    uploaded_file = st.file_uploader("Select YOLO dataset ZIP", type=["zip"], key="dataset_upload")
    ds_name_input = st.text_input("Dataset Name (on server)", key="ds_name_input",
                                   placeholder="e.g. helmet_v1")
    if st.button("Upload Dataset", key="btn_upload") and uploaded_file and ds_name_input.strip():
        try:
            file_bytes = uploaded_file.read()
            result = chunked_upload(api_url, file_bytes, ds_name_input.strip(), uploaded_file.name)
            st.success(f"✅ Uploaded! Server path: `{result['server_path']}`")
            st.session_state.selected_dataset_path = result["server_path"]
            st.session_state.selected_dataset_yaml = result.get("yaml_path", "")
            st.session_state.upload_done = True
            st.rerun()
        except Exception as e:
            st.error(f"Upload failed: {e}")

if dataset_names:
    sel_ds = st.sidebar.selectbox("Select Dataset", dataset_names, key="ds_select")
    ds_info = next((d for d in (datasets_data or []) if d["dataset_name"] == sel_ds), {})
    st.sidebar.caption(f"📂 {ds_info.get('size_mb', '?')} MB | "
                       f"🖼 {ds_info.get('image_count', '?')} images")
    st.session_state.selected_dataset_path = ds_info.get("server_path", "")
    st.session_state.selected_dataset_yaml = ds_info.get("yaml_path", "")
else:
    st.sidebar.info("Upload a dataset first.")

st.sidebar.divider()

# ── Start Research ────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🚀 Start Research")
exp_name       = st.sidebar.text_input("Experiment Name", "Exp_01")
target_acc     = st.sidebar.slider("Target mAP50", 0.1, 0.99, 0.80, 0.01)
max_cycles     = st.sidebar.slider("Max Search Cycles", 1, 10, 3)

can_start = (
    bool(st.session_state.selected_project) and
    bool(st.session_state.selected_dataset_path) and
    not st.session_state.agent_running
)

if st.sidebar.button("▶ Start Autoresearch", disabled=not can_start, type="primary"):
    yaml_path = st.session_state.selected_dataset_yaml or st.session_state.selected_dataset_path
    inputs = {
        "project_name":        st.session_state.selected_project,
        "project_description": proj_info.get("description", ""),
        "experiment_name":     exp_name,
        "dataset_path":        yaml_path,
        "target_accuracy":     target_acc,
        "max_cycles":          max_cycles,
        "api_url":             api_url,
        "mlflow_uri":          mlflow_uri,
        "current_cycle":       0,
        "run_id":              "",
        "data_stats":          {},
        "current_model":       "",
        "current_hyperparams": {},
        "history":             [],
        "status":              "Starting…",
        "error":               None,
    }
    st.session_state.agent_running = True
    st.session_state.agent_state   = inputs
    t = threading.Thread(target=run_workflow_bg, args=(inputs, ollama_url), daemon=True)
    t.start()
    st.sidebar.success("🚀 Research loop started!")

if not can_start and not st.session_state.agent_running:
    missing = []
    if not st.session_state.selected_project:
        missing.append("project")
    if not st.session_state.selected_dataset_path:
        missing.append("dataset")
    if missing:
        st.sidebar.caption(f"⚠️ Select a {' and '.join(missing)} to start.")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🤖 YOLO AutoResearch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Agentic framework — automated model selection, hyperparameter tuning & iterative training</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_loop, tab_mlflow, tab_data, tab_history = st.tabs([
    "🔬 Research Loop", "📈 MLflow", "📦 Data Management", "🗃️ History & Downloads"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Research Loop
# ══════════════════════════════════════════════════════════════════════════════
with tab_loop:
    state = st.session_state.agent_state

    if not state:
        st.info("Configure project & dataset in the sidebar, then click **▶ Start Autoresearch**.")
    else:
        render_stepper(state)

        err = state.get("error")
        if err:
            st.error(f"❌ **Error:** {err}")
        elif st.session_state.agent_running:
            st.info(f"🔄 {state.get('status', 'Running…')}")
        else:
            st.success(f"✅ {state.get('status', 'Completed.')}")

        # Metrics row
        history = state.get("history", [])
        best_map = max((h.get("metrics", {}).get("map50", 0.0) for h in history), default=0.0)
        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, val in [
            (c1, "Cycle", f"{state.get('current_cycle', 0)} / {state.get('max_cycles', '?')}"),
            (c2, "Active Model", f"{state.get('current_model', '—')}.pt" if state.get("current_model") else "—"),
            (c3, "Best mAP50",   f"{best_map:.4f}"),
            (c4, "Target",       f"{state.get('target_accuracy', '?')}"),
        ]:
            col.markdown(f'<div class="card"><div class="metric-lbl">{lbl}</div>'
                         f'<div class="metric-val">{val}</div></div>', unsafe_allow_html=True)

        # Dataset stats
        stats = state.get("data_stats")
        if stats:
            with st.expander("📊 Dataset Analysis", expanded=True):
                ca, cb, cc = st.columns(3)
                with ca:
                    st.write(f"**Classes:** {stats.get('num_classes')}")
                    st.write(f"**Names:** {', '.join(stats.get('class_names', []))}")
                with cb:
                    st.write(f"**Images:** {stats.get('total_images')}")
                    st.write(f"**Boxes scanned:** {stats.get('total_boxes')}")
                with cc:
                    dist = stats.get("bbox_distribution", {})
                    st.write(f"**Small bboxes:** {dist.get('small')}%")
                    st.write(f"**Medium bboxes:** {dist.get('medium')}%")
                    st.write(f"**Large bboxes:** {dist.get('large')}%")

        # Live training monitor
        active_run = state.get("run_id")
        if active_run and (st.session_state.agent_running or state.get("status", "").startswith("Cycle")):
            st.markdown("### 🎛️ Live Training Monitor")
            live, lerr = api_get(api_url, f"/api/train/status/{active_run}", timeout=5)
            if live:
                prog = live.get("progress", {})
                pct  = prog.get("pct", 0.0)
                st.progress(pct / 100.0)
                st.caption(f"Epoch {prog.get('current_epoch')}/{prog.get('total_epochs')} — {pct}%")
                logs = live.get("logs", "")
                if logs:
                    st.markdown(f'<div class="log-box">{logs}</div>', unsafe_allow_html=True)

        # Cycle history
        if history:
            st.markdown("### 🧠 Research Cycle History")
            for h in reversed(history):
                m = h.get("metrics", {})
                with st.expander(
                    f"Cycle {h['cycle']} — {h.get('model', '?')}.pt — mAP50: {m.get('map50', 0.0):.4f}",
                    expanded=(h["cycle"] == state.get("current_cycle"))
                ):
                    st.info(f"**LLM Reasoning:** {h.get('reasoning', '—')}")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**Metrics:**")
                        for mk, mv in m.items():
                            st.write(f"- {mk}: `{mv:.4f}`")
                    with col_b:
                        st.write("**Hyperparameters:**")
                        for hk, hv in h.get("hyperparams", {}).items():
                            st.write(f"- {hk}: `{hv}`")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MLflow
# ══════════════════════════════════════════════════════════════════════════════
with tab_mlflow:
    st.markdown("### 📈 MLflow Experiment Tracking")
    reachable = False
    try:
        requests.get(mlflow_uri, timeout=3)
        reachable = True
    except Exception:
        pass

    if reachable:
        st.markdown(f"""
        <iframe src="{mlflow_uri}" width="100%" height="820"
          style="border:1px solid #2d2d4a;border-radius:10px;"></iframe>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ MLflow not reachable at `{mlflow_uri}`.")
        st.code(f"mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Data Management
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### 📦 Datasets on Training Server")
    if st.button("🔄 Refresh", key="refresh_datasets"):
        st.rerun()

    ds_list, ds_err = api_get(api_url, "/api/data/list")
    if ds_err:
        st.error(f"Cannot fetch datasets: {ds_err}")
    elif not ds_list:
        st.info("No datasets uploaded yet. Use the sidebar to upload a YOLO dataset ZIP.")
    else:
        for ds in ds_list:
            with st.container():
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.markdown(f"**📁 {ds['dataset_name']}**")
                    st.caption(f"Path: `{ds['server_path']}`")
                    if ds.get("yaml_path"):
                        st.caption(f"YAML: `{ds['yaml_path']}`")
                with col2:
                    st.metric("Size", f"{ds.get('size_mb', '?')} MB")
                    st.metric("Images", ds.get("image_count", "?"))
                with col3:
                    if st.button("🗑️ Delete", key=f"del_{ds['dataset_name']}"):
                        _, err = api_delete(api_url, f"/api/data/{ds['dataset_name']}")
                        if err:
                            st.error(err)
                        else:
                            st.success("Deleted.")
                            st.rerun()
                st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — History & Downloads
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 🗃️ All Training Runs")
    if st.button("🔄 Refresh Jobs", key="refresh_jobs"):
        st.rerun()

    jobs, jobs_err = api_get(api_url, "/api/jobs")
    if jobs_err:
        st.error(f"Cannot reach API: {jobs_err}")
    elif not jobs:
        st.info("No training jobs recorded yet.")
    else:
        rows = []
        for j in jobs:
            m = {}
            if j.get("metrics"):
                try:
                    m = json.loads(j["metrics"])
                except Exception:
                    pass
            rows.append({
                "Run ID":     j["run_id"],
                "Project":    j["project_name"],
                "Experiment": j["experiment_name"],
                "Model":      j["model_scale"],
                "Status":     j["status"].upper(),
                "mAP50":      round(m.get("map50", 0.0), 4),
                "Precision":  round(m.get("precision", 0.0), 4),
                "Recall":     round(m.get("recall", 0.0), 4),
                "Created":    j["created_at"][:19],
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### ⬇️ Download Artifacts")
        run_ids = [j["run_id"] for j in jobs]
        sel_run = st.selectbox("Select run", run_ids, key="dl_run_sel")
        sel_job = next((j for j in jobs if j["run_id"] == sel_run), None)
        if sel_job:
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Project:** {sel_job['project_name']}")
                st.write(f"**Experiment:** {sel_job['experiment_name']}")
                st.write(f"**Status:** {sel_job['status'].upper()}")
            with c2:
                dl_url = f"{api_url}/api/train/download/{sel_run}"
                st.markdown(
                    f'<a href="{dl_url}" target="_blank">'
                    f'<button style="background:#6C63FF;color:#fff;border:none;'
                    f'padding:10px 22px;border-radius:8px;font-weight:600;cursor:pointer;'
                    f'font-size:.9rem;">⬇️ Download ZIP</button></a>',
                    unsafe_allow_html=True
                )
                st.caption("Includes: `weights/best.pt`, `weights/last.pt`, `results.csv`, charts")

            if sel_job["status"] == "running":
                if st.button("⏹ Stop Training", key="btn_stop"):
                    res, err = api_post(api_url, f"/api/train/stop/{sel_run}")
                    if err:
                        st.error(err)
                    else:
                        st.warning("Training stopped.")
                        st.rerun()

# ── Auto-refresh while agent is running ───────────────────────────────────────
if st.session_state.agent_running:
    time.sleep(3)
    st.rerun()
