import yaml
import uvicorn
import os

def main():
    config_path = "config/settings.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}. Starting with defaults.")
        host = "0.0.0.0"
        port = 8000
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        server_cfg = config.get("server", {})
        host = server_cfg.get("host", "0.0.0.0")
        port = server_cfg.get("port", 8000)
        
    print(f"Starting server on {host}:{port}...")
    uvicorn.run("server.main:app", host=host, port=port)

if __name__ == "__main__":
    main()
