from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import Dict, Any
import os


class Settings(BaseSettings):
    server_script_path: str = os.path.join("MCP", "server.py")

class QueryRequest(BaseModel):
    query: str

class Message(BaseModel):
    role: str
    content: Any

class ToolCall(BaseModel):
    name: str
    args: Dict[Any, Any]

