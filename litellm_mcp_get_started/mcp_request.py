# Create server parameters for stdio connection
import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

server_params = {
    "url": "http://localhost:4000/mcp/dwhttp",
    "headers": {
        "Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}"
    }
}

async def main():
    async with streamablehttp_client(**server_params) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Load the remote graph as if it was a tool
            tools = await load_mcp_tools(session)

            # Create and run a react agent with the tools
            agent = create_agent("openai:openai/gpt-4.1", tools)

            # Invoke the agent with a message
            agent_response = await agent.ainvoke({"messages": "你可以使用 deepwiki 工具來回答我關於 Python 程式設計的問題嗎？"})
            print(agent_response)

if __name__ == "__main__":
    asyncio.run(main())