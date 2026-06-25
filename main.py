import os
import sys
import subprocess
import argparse
import time

def get_venv_bin(bin_name: str) -> str:
    """
    Returns the path to a binary inside the same virtualenv as the running python interpreter.
    """
    venv_bin_dir = os.path.dirname(sys.executable)
    local_path = os.path.join(venv_bin_dir, bin_name)
    if os.path.exists(local_path):
        return local_path
    return bin_name # Fallback to system PATH

def start_api():
    print("Starting FastAPI Training Server...")
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

def start_frontend():
    print("Starting Streamlit Frontend Workspace...")
    streamlit_bin = get_venv_bin("streamlit")
    cmd = [
        streamlit_bin, "run", "frontend/app.py", 
        "--server.address", "0.0.0.0", 
        "--server.port", "8501"
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("Stopping Streamlit...")

def start_mlflow():
    print("Starting MLflow Tracking Server...")
    mlflow_bin = get_venv_bin("mlflow")
    cmd = [
        mlflow_bin, "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns"
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("Stopping MLflow...")

def start_all():
    print("=========================================")
    print("Starting YOLO AutoResearch Workspace")
    print("=========================================")
    
    mlflow_bin = get_venv_bin("mlflow")
    streamlit_bin = get_venv_bin("streamlit")
    
    processes = []
    try:
        # 1. Start MLflow Server
        mlflow_cmd = [
            mlflow_bin, "server",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "./mlruns"
        ]
        mlflow_proc = subprocess.Popen(mlflow_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(("MLflow", mlflow_proc))
        print("✓ Started MLflow Server (http://localhost:5000)")
        
        # 2. Start FastAPI Server
        api_cmd = [
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]
        api_proc = subprocess.Popen(api_cmd)
        processes.append(("FastAPI Backend", api_proc))
        print("✓ Started FastAPI Backend (http://localhost:8000)")
        
        # Give backend & MLflow a second to spin up
        time.sleep(2)
        
        # 3. Start Streamlit Frontend
        frontend_cmd = [
            streamlit_bin, "run", "frontend/app.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501"
        ]
        frontend_proc = subprocess.Popen(frontend_cmd)
        processes.append(("Streamlit Frontend", frontend_proc))
        print("✓ Started Streamlit Frontend (http://localhost:8501)")
        
        print("\nAll servers are running. Press Ctrl+C to terminate all services.")
        
        while True:
            # Check if any process died
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"⚠️ Process {name} terminated with code {proc.returncode}")
                    raise KeyboardInterrupt
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nTerminating all services...")
        for name, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=3)
                print(f"✓ Stopped {name}")
            except Exception as e:
                print(f"Failed to stop {name}: {e}")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Autoresearch Runner CLI")
    parser.add_argument(
        "service", 
        choices=["api", "frontend", "mlflow", "start-all"], 
        help="Specify which service to start"
    )
    
    args = parser.parse_args()
    
    if args.service == "api":
        start_api()
    elif args.service == "frontend":
        start_frontend()
    elif args.service == "mlflow":
        start_mlflow()
    elif args.service == "start-all":
        start_all()
