# core/graph.py
import asyncio
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from state import ResearchState
from nodes import ResearchNodes
from edges import ResearchRouter
from typing import Dict, Literal, Any


class YOLOResearchGraph:
    def __init__(self, ollama_model: str = "qwen3.5:0.8b", checkpoint_db: str = "research_checkpoints.db"):
        # Initialize LLM
        self.llm = ChatOllama(
            model=ollama_model,
            temperature=0.7,
            num_predict=2048
        )
        
        # Initialize nodes and router
        self.nodes = ResearchNodes(self.llm)
        self.router = ResearchRouter()
        
        # Set up checkpointing for persistence
        # self.memory = SqliteSaver.from_conn_string(checkpoint_db)
        self.memory = MemorySaver()
        
        # Build the graph
        self.graph = self._build_graph()


    async def _fetch_mcp_tools(self) -> Dict[str, Any]:
        # Initialize MCP client
        server_params = StdioServerParameters(
            command="python",
            # Make sure to update to the full absolute path to your math_server.py file
            args=["../mcp_server/server.py"],
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                # Get tools
                tools = await load_mcp_tools(session)
                print("MCP Tools fetched successfully\n", tools)
        return tools

    def _build_graph(self):
        """Constructs the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("hypothesize", self.nodes.hypothesis_generator)
        workflow.add_node("edit_code", self.nodes.code_editor)
        mcp_tools = asyncio.run(self._fetch_mcp_tools())
        tools = ToolNode(tools=mcp_tools)
        workflow.add_node("tools", tools)
        workflow.add_node("execute", self.nodes.experiment_executor)
        workflow.add_node("evaluate", self.nodes.metrics_evaluator)
        workflow.add_node("recover", self._recovery_node)
        workflow.add_node("finish", self._finish_node)
        
        # Set entry point
        workflow.set_entry_point("hypothesize")
        
        # Add edges
        workflow.add_edge("hypothesize", "edit_code")
        
        # Conditional edge after code editing
        workflow.add_conditional_edges(
            "edit_code",
            self.router.after_code_edit,
            {
                "execute": "execute",
                "hypothesize": "hypothesize"
            }
        )
        
        # Execute node goes to tools
        workflow.add_edge("execute", "tools")
        
        # Conditional edge after tools
        workflow.add_conditional_edges(
            "tools",
            self.router.after_tool_call,
            {
                "evaluate": "evaluate",
                "tools": "tools",
                "__end__": END
            }
        )
        
        # After evaluation, decide whether to continue
        workflow.add_conditional_edges(
            "evaluate",
            self.router.should_continue_research,
            {
                "hypothesize": "hypothesize",
                "finish": "finish",
                "recover": "recover"
            }
        )
        
        # Recovery can lead back to hypothesize or finish
        workflow.add_conditional_edges(
            "recover",
            self._recovery_router,
            {
                "hypothesize": "hypothesize",
                "finish": "finish"
            }
        )
        
        workflow.add_edge("finish", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    async def _recovery_node(self, state: ResearchState) -> Dict[str, Any]:
        """
        Handles recovery when experiments are failing or stagnating.
        This might reset certain parameters or change strategy.
        """
        recovery_prompt = f"""
        We've encountered issues in our research:
        - Consecutive failures: {state['consecutive_failures']}
        - Last error: {state.get('last_error', 'None')}
        
        Should we:
        1. Roll back to the best model and try a completely different approach?
        2. Reduce complexity of changes?
        3. Stop research and report best results?
        
        Respond with a structured decision.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=recovery_prompt)])
        
        # Reset failure counters if continuing
        return {
            "messages": [response],
            "consecutive_failures": 0,
            "recovery_attempts": state.get('recovery_attempts', 0) + 1
        }
    
    async def _finish_node(self, state: ResearchState) -> Dict[str, Any]:
        """
        Final node that prepares the research summary.
        """
        summary_prompt = f"""
        Research completed! Summarize the findings:
        - Best metrics achieved: {state['best_metrics']}
        - Total experiments run: {len(state['experiment_history'])}
        - Best model path: {state['best_model_path']}
        - Key insights from successful experiments
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
        
        return {
            "messages": [response],
            "should_continue": False
        }
    
    def _recovery_router(self, state: ResearchState) -> Literal["hypothesize", "finish"]:
        """Decides whether to continue after recovery"""
        if state.get('recovery_attempts', 0) >= 3:
            return "finish"
        return "hypothesize"
    
    async def run_research(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point to run the research workflow.
        """
        config = {"configurable": {"thread_id": initial_state.get("thread_id", "default")}}
        
        final_state = await self.graph.ainvoke(initial_state, config)
        return final_state
    
if __name__ == "__main__":
    from IPython.display import Image, display

    researcher = YOLOResearchGraph()
    img = Image(researcher.graph.get_graph().draw_mermaid_png())
    # display()
    with open('sample.jpg', 'wb') as f:
        f.write(img.data)