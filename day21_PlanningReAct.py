import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


async def run_trip_planner():
    # 從環境變數讀取 AccuWeather API Key
    accuweather_api_key = os.getenv("ACCUWEATHER_API_KEY")
    if not accuweather_api_key:
        raise ValueError("請設定 ACCUWEATHER_API_KEY 環境變數")
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "stdio",
                "command": "uv",
                "args": ["--directory", "/Users/juishu/src/weather", "run", "weather_mcp_server.py"],
                "env": {"ACCUWEATHER_API_KEY": accuweather_api_key}
            },
            "attractions": {
                "transport": "stdio",
                "command": "uv",
                "args": ["--directory", "/Users/juishu/src/wikivoyage", "run", "attractions_mcp_server.py"]
            }
        }
    )
    tools = await client.get_tools()

    MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
    LLM = init_chat_model(
        MODEL_NAME,
        temperature=0.2,
    )

    agent = create_agent(
        model=LLM,
        tools=tools,
        system_prompt=(
            "你是一位智慧旅遊助理，要先規劃維也納一日行程（早上、下午、晚上）。"
            "每個時段都需檢查景點是否營業（使用 check_open_status 工具）。"
            "若景點關閉，請使用 get_attractions 重新挑選並更新行程。"
            "請在過程中輸出使用的工具與輸入參數。"
        )
    )

    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "幫我規劃今天的維也納一日遊"}]
    })

    print("\n=== Agent 推理與規劃歷程 ===")
    for m in response["messages"]:
        if hasattr(m, "tool_calls") and m.tool_calls:
            print(f"🧰 {m.tool_calls}")
        else:
            print(f"💭 {getattr(m, 'content', '')}")

    print("\n=== 最終建議 ===")
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(run_trip_planner())