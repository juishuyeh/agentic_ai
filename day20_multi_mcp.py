import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def run_agent_with_multi_mcp():
    # 從環境變數讀取 AccuWeather API Key
    accuweather_api_key = os.getenv("ACCUWEATHER_API_KEY")
    if not accuweather_api_key:
        raise ValueError("請設定 ACCUWEATHER_API_KEY 環境變數")
    
    # 建立 MultiServerMCPClient，並設定 AccuWeather MCP Server
    client = MultiServerMCPClient(
        {
            "accuweather": {
                "transport": "stdio",
                "command": "uv",
                "args": [
                    "--directory",
                    "/Users/juishu/src/weather",
                    "run",
                    "weather_mcp_server.py"
                ],
                "env": {"ACCUWEATHER_API_KEY": accuweather_api_key}
            },
            "attractions": {
                "transport": "stdio",
                "command": "uv",
                "args": [
                    "--directory",
                    "/Users/juishu/src/wikivoyage",
                    "run",
                    "attractions_mcp_server.py"
                ]
            }
        }
    )

    # 取得 MCP 工具
    tools = await client.get_tools()
    
    MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
    LLM = init_chat_model(
        MODEL_NAME,
        temperature=0.2,
        max_tokens=2048,
    )
    # 建立 Agent，並加入 MCP 工具
    agent = create_agent(
        model=LLM,
        tools=tools,
        system_prompt="你是一位旅遊助理，可以根據天氣與景點資料，推薦最合適的維也納旅遊行程。"
    )

    # 與 Agent 對話，詢問維也納的推薦景點
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "幫我依今天天氣推薦維也納景點"}]
    })

    print("\n=== 對話歷程 ===")
    for m in response["messages"]:
        role = m.__class__.__name__
        print(f"[{role}] {getattr(m, 'content', '') or getattr(m, 'tool_calls', '')}")

    print("\n=== 模型最終回答 ===")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(run_agent_with_multi_mcp())