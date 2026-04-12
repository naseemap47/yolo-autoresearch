import streamlit as st
from typing import Dict, Any
import httpx


class Chatbot:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.messages = st.session_state["messages"]

    async def get_tools(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=100) as client:
            response = await client.get(f"{self.api_url}/tools")
            return response.json()

    def display_message(self, message: Dict[str, Any]):
        # User Message
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        # Tool Use Message
        if "thinking" in message:
            st.chat_message("assistant").write(message["thinking"])
            for tool in message["tools"]:
                st.chat_message("assistant").json({
                    "name": tool["function"]["name"],
                    "arguments": tool["function"]["arguments"],
                }, expanded=False)

        # Tool Result Message
        if message["role"] == "tool":
            st.chat_message("assistant").json({
                "result": message["content"]
            }, expanded=False)

        # AI Message
        if (message["role"] == "assistant") and "content" in message:
            st.chat_message("assistant").write(message["content"])

    async def render(self):
        st.title("MCP Client")
        with st.sidebar:
            st.subheader("Settings")
            st.write(f"API URL: {self.api_url}")
            result = await self.get_tools()
            st.subheader("Tools")
            st.write([tool["name"] for tool in result["tools"]])
        
        query = st.chat_input("Ask any question")
        if query:
            st.chat_message("user").write(query)
            async with httpx.AsyncClient(timeout=600) as client:
                response = await client.post(f"{self.api_url}/query", json={"query": query})
                if response.status_code == 200:
                    messages = response.json()["messages"]
                    st.session_state["messages"] = messages
                    for message in messages:
                        # st.chat_message(message["role"]).write(message["content"])
                        self.display_message(message)
