# core/edges.py
from typing import Literal
from state import ResearchState


class ResearchRouter:
    @staticmethod
    def should_continue_research(state: ResearchState) -> Literal["hypothesize", "finish", "recover"]:
        """
        Decides the next step in the research workflow.
        This is called after each experiment evaluation.
        """
        # Check for fatal errors
        if state.get('consecutive_failures', 0) >= 3:
            return "recover"
        
        # Check if we've reached max iterations
        if state['iteration'] >= state['max_iterations']:
            return "finish"
        
        # Check if we've reached target metric
        current_best = state['best_metrics'].get(state['target_metric'], 0)
        if current_best >= state['target_threshold']:
            return "finish"
        
        # Check for stagnation (no improvement in last 5 experiments)
        if len(state['experiment_history']) >= 5:
            last_5_metrics = [
                exp.metrics.get(state['target_metric'], 0) 
                for exp in state['experiment_history'][-5:]
                if exp.success
            ]
            if last_5_metrics and max(last_5_metrics) <= current_best * 1.01:  # < 1% improvement
                # Stagnation detected - maybe try a different approach
                return "recover"
        
        # Continue research
        return "hypothesize"

    @staticmethod
    def after_code_edit(state: ResearchState) -> Literal["execute", "hypothesize"]:
        """
        Routes after code editing. If the LLM produced valid changes,
        execute them. Otherwise, go back to hypothesis generation.
        """
        if state.get('pending_code_changes'):
            return "execute"
        return "hypothesize"

    @staticmethod
    def after_tool_call(state: ResearchState) -> Literal["evaluate", "tools", "__end__"]:
        """
        Standard LangGraph routing for tool usage.
        """
        last_message = state['messages'][-1]
        
        # If the last message has tool calls, route to the tool node
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # If we have experiment results, evaluate them
        if state.get('current_experiment'):
            return "evaluate"
        
        # Otherwise, end
        return "__end__"