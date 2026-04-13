from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import Dict, Any
import os


class Settings(BaseSettings):
    server_script_path: str = os.path.join("mcp_server", "server.py")

class QueryRequest(BaseModel):
    query: str

class Message(BaseModel):
    role: str
    content: Any

class ToolCall(BaseModel):
    name: str
    args: Dict[Any, Any]

class TrainRequest(BaseModel):
    model_name: str
    data_path: str
    epochs: int
    patience: int
    batch: int | float
    imgsz: int
    device: int | str | list
    name: str
    optimizer: str
    single_cls: bool
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float