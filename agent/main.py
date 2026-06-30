import argparse
import yaml
import os
from .graph import create_agent_graph
from .rag import get_or_build_vectorstore, UltralyticsRetriever

def main():
    parser = argparse.ArgumentParser(description="YOLO Auto-Research Agent")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to config file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found at {args.config}. Please create it.")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    agent_cfg = config.get("agent", {})
    gpu_host = agent_cfg.get("gpu_host", "192.168.0.84")
    llm_model = agent_cfg.get("llm_model", "qwen3.5:9b")
    target_map = agent_cfg.get("target_map", 0.8)
    max_cycles = agent_cfg.get("max_cycles", 5)
    dataset_yaml_path = agent_cfg.get("dataset_yaml_path")

    if not dataset_yaml_path:
        print("dataset_yaml_path is required in config.")
        return

    # ------------------------------------------------------------------ #
    # MLflow Configuration                                                 #
    # ------------------------------------------------------------------ #
    mlflow_cfg = config.get("mlflow", {})
    if mlflow_cfg:
        tracking_uri    = mlflow_cfg.get("tracking_uri", "http://127.0.0.1:5000")
        experiment_name = mlflow_cfg.get("experiment_name", "yolo-autoresearch")
        print(f"[MLflow] Tracking URI : {tracking_uri}")
        print(f"[MLflow] Experiment   : {experiment_name}")
        print(f"[MLflow] Open the UI  : mlflow ui --port 5000  →  {tracking_uri}")
    else:
        print("[MLflow] No 'mlflow' section in config — running without experiment tracking.")

    # ------------------------------------------------------------------ #
    # RAG Initialization                                                   #
    # ------------------------------------------------------------------ #
    rag_cfg = config.get("rag", {})
    retriever = None

    if rag_cfg:
        docs_path      = rag_cfg.get("docs_path", "finetune/ultralytics_raw.txt")
        persist_dir    = rag_cfg.get("persist_dir", "agent/vectorstore")
        embed_model    = rag_cfg.get("embedding_model", "nomic-embed-text")
        top_k          = rag_cfg.get("top_k", 4)
        chunk_size     = rag_cfg.get("chunk_size", 800)
        chunk_overlap  = rag_cfg.get("chunk_overlap", 100)

        try:
            vectorstore = get_or_build_vectorstore(
                docs_path=docs_path,
                persist_dir=persist_dir,
                embedding_model=embed_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            retriever = UltralyticsRetriever(vectorstore=vectorstore, top_k=top_k)
            print(f"[RAG] Ready — top_k={top_k}, embed_model={embed_model}")
        except Exception as e:
            print(f"[RAG] WARNING: Could not initialize RAG ({e}). Running without context.")
            retriever = None
    else:
        print("[RAG] No 'rag' section found in config — running without documentation context.")

    # ------------------------------------------------------------------ #
    # Agent Graph                                                          #
    # ------------------------------------------------------------------ #
    agent = create_agent_graph(
        host=gpu_host,
        model_name=llm_model,
        retriever=retriever,
        target_map=target_map,
        mlflow_cfg=mlflow_cfg,
    )

    initial_state = {
        "dataset_yaml_path": dataset_yaml_path,
        "target_map": target_map,
        "max_cycles": max_cycles,
        "current_cycle": 0,
        "dataset_stats": {},
        "history": [],
        "current_config": {},
        "final_result": ""
    }

    print("Starting YOLO Auto-Research Agent...")
    print(f"Target mAP: {target_map}, Max Cycles: {max_cycles}, LLM: {llm_model}, Host: {gpu_host}")

    final_state = agent.invoke(initial_state)

    print("\n--- Auto-Research Finished ---")
    print(final_state["final_result"])
    print("\nFinal History:")
    for record in final_state["history"]:
        print(f"Cycle {record['cycle']}: Config={record['config']} -> mAP50-95={record['results'].get('mAP50-95')}")

if __name__ == "__main__":
    main()
