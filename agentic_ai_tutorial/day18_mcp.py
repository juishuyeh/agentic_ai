import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def run_agent_with_accuweather_mcp():
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
            }
        }
    )

    # 取得 MCP 工具
    tools = await client.get_tools()

    
    # 建立 Gemini-2.5-FLASH LLM

    # =============== Step 0: 初始化模型 ===============
    MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
    LLM = init_chat_model(
        MODEL_NAME,
        temperature=0.2,
    )

    # 建立 Agent，並加入 MCP 工具
    agent = create_agent(
        model=LLM,
        tools=tools,
        system_prompt="你是一位天氣助理，可以透過工具查詢城市的即時天氣。"
    )

    # 與 Agent 對話，詢問維也納的天氣
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "維也納的天氣如何？"}]
    })

    print("\n=== 對話歷程 ===")
    for m in response["messages"]:
        role = m.__class__.__name__
        print(f"[{role}] {getattr(m, 'content', '') or getattr(m, 'tool_calls', '')}")

    print("\n=== 模型最終回答 ===")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(run_agent_with_accuweather_mcp())