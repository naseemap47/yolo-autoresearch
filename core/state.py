# core/state.py
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph.message import add_messages
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ExperimentResult:
    """Tracks a single experiment run"""
    run_id: str
    timestamp: datetime
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]  # mAP50, mAP50-95, precision, recall
    code_changes: str  # Git diff of changes made
    hypothesis: str
    success: bool
    model_path: str

class ResearchState(TypedDict):
    # === Conversation Memory ===
    messages: Annotated[List[Dict], add_messages]  # Chat history with Ollama
    
    # === Research Progress ===
    iteration: int
    research_goal: str
    dataset_info: Dict[str, Any]  # num_classes, image_size, class_names
    
    # === Model Tracking ===
    base_model: str  # 'yolov8n.pt', 'yolov8m.pt', etc.
    current_model_path: str
    best_model_path: str
    best_metrics: Dict[str, float]
    
    # === Experiment History ===
    experiment_history: List[ExperimentResult]
    failed_attempts_count: int
    consecutive_failures: int
    
    # === Current Experiment ===
    current_hypothesis: str
    pending_code_changes: str
    current_experiment: Optional[ExperimentResult]
    
    # === Control Flow ===
    should_continue: bool
    max_iterations: int
    target_metric: str  # e.g., 'metrics/mAP50-95(B)'
    target_threshold: float
    
    # === Error Handling ===
    last_error: Optional[str]
    recovery_attempts: int