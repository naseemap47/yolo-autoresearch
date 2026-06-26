import argparse
import yaml
import os
from .graph import create_agent_graph

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
    llm_model = agent_cfg.get("llm_model", "qwen3.5:0.8b")
    target_map = agent_cfg.get("target_map", 0.8)
    max_cycles = agent_cfg.get("max_cycles", 5)
    dataset_yaml_path = agent_cfg.get("dataset_yaml_path")

    if not dataset_yaml_path:
        print("dataset_yaml_path is required in config.")
        return

    agent = create_agent_graph(host=gpu_host, model_name=llm_model)
    
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
