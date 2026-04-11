# core/nodes.py
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
from typing import Dict, Any, List
from state import ResearchState, ExperimentResult
from datetime import datetime
import uuid


class ResearchNodes:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        
    async def hypothesis_generator(self, state: ResearchState) -> Dict[str, Any]:
        """
        Analyzes past experiments and generates a new hypothesis.
        This is the "thinking" node where Ollama reasons about what to try next.
        """
        system_prompt = """You are an expert YOLO model researcher. Your task is to analyze 
        experiment history and propose ONE specific, testable hypothesis to improve model performance.
        
        Available hyperparameters to adjust:
        - Learning rate (lr0, lrf)
        - Batch size
        - Image size
        - Optimizer (AdamW, SGD)
        - Augmentation parameters (hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, perspective, flipud, fliplr, mosaic, mixup, copy_paste)
        - Loss weights (box, cls, dfl)
        - Architecture modifications (depth_multiple, width_multiple)
        
        Respond with a JSON containing:
        {
            "hypothesis": "Detailed reasoning for the change",
            "code_changes": "Specific YAML or Python code modifications",
            "expected_impact": "What metric will improve and why",
            "risk_level": "low/medium/high"
        }
        """
        
        # Build context from history
        history_summary = self._summarize_history(state['experiment_history'])
        current_best = state.get('best_metrics', {})
        
        prompt = f"""
        Research Goal: {state['research_goal']}
        Current Iteration: {state['iteration']}/{state['max_iterations']}
        Best Metrics So Far: {json.dumps(current_best, indent=2)}
        
        Experiment History Summary:
        {history_summary}
        
        Failed Attempts: {state['failed_attempts_count']}
        Consecutive Failures: {state['consecutive_failures']}
        
        Based on this data, what specific change should we try next?
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        
        # Parse the response
        try:
            hypothesis_data = json.loads(response.content)
            return {
                "messages": [response],
                "current_hypothesis": hypothesis_data["hypothesis"],
                "pending_code_changes": hypothesis_data["code_changes"],
            }
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return proper JSON
            return {
                "messages": [response],
                "current_hypothesis": response.content,
                "pending_code_changes": "",
            }

    async def code_editor(self, state: ResearchState) -> Dict[str, Any]:
        """
        Prepares the MCP tool call to modify the training configuration.
        This doesn't execute the change yet - just prepares the tool call.
        """
        # Create a tool call message for the MCP server
        tool_call_msg = AIMessage(
            content=f"Preparing to apply changes: {state['current_hypothesis']}",
            tool_calls=[{
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": "modify_training_config",
                "args": {
                    "config_changes": state['pending_code_changes'],
                    "base_model": state['base_model'],
                    "experiment_name": f"exp_{state['iteration']:03d}"
                }
            }]
        )
        
        return {
            "messages": [tool_call_msg],
        }

    async def experiment_executor(self, state: ResearchState) -> Dict[str, Any]:
        """
        Executes the training using MCP tool and captures results.
        This node runs AFTER the tool has been called by the ToolNode.
        """
        # The actual execution happens through the ToolNode
        # This node processes the results
        last_message = state['messages'][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Tool was called, now we need to wait for execution
            tool_result_msg = AIMessage(
                content="Experiment execution initiated",
                tool_calls=[{
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": "train_yolo_model",
                    "args": {
                        "experiment_name": f"exp_{state['iteration']:03d}",
                        "epochs": 100,
                        "patience": 20,
                        "save_period": 10
                    }
                }]
            )
            return {"messages": [tool_result_msg]}
        
        return {}

    async def metrics_evaluator(self, state: ResearchState) -> Dict[str, Any]:
        """
        Evaluates the results of the experiment after training completes.
        """
        # Extract metrics from the tool response
        last_message = state['messages'][-1]
        
        try:
            # Assuming the MCP tool returns structured metrics
            metrics = json.loads(last_message.content)
            
            experiment = ExperimentResult(
                run_id=f"run_{state['iteration']:03d}",
                timestamp=datetime.now(),
                hyperparameters=metrics.get('hyperparameters', {}),
                metrics=metrics.get('results', {}),
                code_changes=state['pending_code_changes'],
                hypothesis=state['current_hypothesis'],
                success=True,
                model_path=metrics.get('model_path', '')
            )
            
            # Determine if this is the best model
            current_metric = metrics['results'].get(state['target_metric'], 0)
            best_metric = state['best_metrics'].get(state['target_metric'], 0)
            
            is_best = current_metric > best_metric
            
            return {
                "current_experiment": experiment,
                "experiment_history": state['experiment_history'] + [experiment],
                "best_metrics": metrics['results'] if is_best else state['best_metrics'],
                "best_model_path": experiment.model_path if is_best else state['best_model_path'],
            }
            
        except Exception as e:
            # Handle failed experiments
            failed_experiment = ExperimentResult(
                run_id=f"run_{state['iteration']:03d}",
                timestamp=datetime.now(),
                hyperparameters={},
                metrics={},
                code_changes=state['pending_code_changes'],
                hypothesis=state['current_hypothesis'],
                success=False,
                model_path=""
            )
            
            return {
                "current_experiment": failed_experiment,
                "experiment_history": state['experiment_history'] + [failed_experiment],
                "failed_attempts_count": state['failed_attempts_count'] + 1,
                "consecutive_failures": state['consecutive_failures'] + 1,
                "last_error": str(e)
            }

    def _summarize_history(self, history: List[ExperimentResult]) -> str:
        """Creates a concise summary of experiment history for the LLM"""
        if not history:
            return "No experiments run yet."
        
        summary = []
        for exp in history[-5:]:  # Last 5 experiments
            summary.append(
                f"- {exp.run_id}: {exp.hypothesis[:100]}... "
                f"mAP: {exp.metrics.get('metrics/mAP50-95(B)', 'N/A')} "
                f"({'Success' if exp.success else 'Failed'})"
            )
        return "\n".join(summary)