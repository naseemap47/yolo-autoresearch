import argparse
from .graph import create_agent_graph

def main():
    parser = argparse.ArgumentParser(description="YOLO Auto-Research Agent")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml on the remote GPU server")
    parser.add_argument("--host", type=str, default="192.168.0.84", help="Remote GPU server IP")
    parser.add_argument("--target-map", type=float, default=0.8, help="Target mAP50-95 threshold")
    parser.add_argument("--max-cycles", type=int, default=5, help="Maximum number of training cycles")
    parser.add_argument("--llm", type=str, default="qwen3.5:0.8b", help="Ollama model to use")
    args = parser.parse_args()

    agent = create_agent_graph(host=args.host, model_name=args.llm)
    
    initial_state = {
        "dataset_yaml_path": args.data,
        "target_map": args.target_map,
        "max_cycles": args.max_cycles,
        "current_cycle": 0,
        "dataset_stats": {},
        "history": [],
        "current_config": {},
        "final_result": ""
    }
    
    print("Starting YOLO Auto-Research Agent...")
    print(f"Target mAP: {args.target_map}, Max Cycles: {args.max_cycles}")
    
    final_state = agent.invoke(initial_state)
    
    print("\n--- Auto-Research Finished ---")
    print(final_state["final_result"])
    print("\nFinal History:")
    for record in final_state["history"]:
        print(f"Cycle {record['cycle']}: Config={record['config']} -> mAP50-95={record['results'].get('mAP50-95')}")

if __name__ == "__main__":
    main()
