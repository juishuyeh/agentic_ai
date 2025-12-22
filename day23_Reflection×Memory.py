import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

load_dotenv()

# 共享狀態
class TripState(dict):
    plan: str | None = None
    weather: str | None = None
    reflection: str | None = None
    memory_summary: str | None = None
    output: str | None = None


# 初始化 MCP 工具
async def init_tools():
    accuweather_api_key = os.getenv("ACCUWEATHER_API_KEY")
    if not accuweather_api_key:
        raise ValueError("請設定 ACCUWEATHER_API_KEY 環境變數")
    
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
    tools = await client.get_tools()
    return tools


# 建立 Agent（啟用短期記憶）
async def build_agent(tools):
    MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
    LLM = init_chat_model(MODEL_NAME, temperature=0.2)

    checkpointer = InMemorySaver()
    return create_agent(
        model=LLM,
        tools=tools,
        checkpointer=checkpointer,
        system_prompt=(
            "你是一位智慧旅遊助理。可透過 MCP 工具查天氣、查景點、檢查營業時段，"
            "並能根據反思結果更新決策。回覆時請先思考所需資訊，再選擇適當工具與參數。"
        )
    )


# 主程式：LangGraph 串接各節點
async def main():
    tools = await init_tools()
    agent = await build_agent(tools)

    # 規劃行程
    async def step_plan(state: TripState):
        print("[PLAN] 規劃初步行程")
        res = await agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": "請規劃維也納一日行程（上午、下午、晚上），並在需要時自行查詢景點。"
            }]
        }, {"configurable": {"thread_id": "1"}})
        state["plan"] = res["messages"][-1].content
        return state

    # 查天氣
    async def step_weather(state: TripState):
        print("[WEATHER] 查天氣並檢視影響")
        res = await agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"以下是目前行程：\n{state['plan']}\n"
                           "請查維也納今天天氣，並分析是否需要調整行程。"
            }]
        }, {"configurable": {"thread_id": "1"}})
        state["weather"] = res["messages"][-1].content
        return state

    # 反思
    async def step_reflect(state: TripState):
        print("[REFLECT] 檢討並提出改進準則")
        res = await agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": (
                    "請基於行程與天氣、營業時段等可得資訊，指出不合理之處並提出改進準則。"
                    "輸出請包含：問題點、原因、改進原則三項。"
                )
            }]
        }, {"configurable": {"thread_id": "1"}})
        state["reflection"] = res["messages"][-1].content
        print("[REFLECT] 結果：")
        print(state["reflection"])
        return state

    # 記憶摘要
    async def step_memory_update(state: TripState):
        print("[MEMORY] 將反思摘要為記憶")
        res = await agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"請將以下反思整理成可重用的決策準則摘要：\n{state['reflection']}"
            }]
        }, {"configurable": {"thread_id": "1"}})
        state["memory_summary"] = res["messages"][-1].content
        print("[MEMORY] 摘要：")
        print(state["memory_summary"])
        return state

    # 重新生成行程
    async def summarize(state: TripState):
        print("[SUMMARY] 基於記憶輸出改良行程")
        res = await agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": (
                    "請基於本次的決策準則摘要，重新輸出更合適的維也納一日行程。"
                    "若需要，請自行再次查詢景點或營業時段。"
                )
            }]
        }, {"configurable": {"thread_id": "1"}})
        state["output"] = res["messages"][-1].content
        print("[SUMMARY] 最終輸出：")
        print(state["output"])
        return state

    # LangGraph 流程設定
    graph = StateGraph(TripState)
    graph.add_node("plan", step_plan)
    graph.add_node("weather", step_weather)
    graph.add_node("reflect", step_reflect)
    graph.add_node("memory", step_memory_update)
    graph.add_node("summary", summarize)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "weather")
    graph.add_edge("weather", "reflect")
    graph.add_edge("reflect", "memory")
    graph.add_edge("memory", "summary")
    graph.add_edge("summary", END)

    app = graph.compile()
    result = await app.ainvoke(TripState())

    print("\n========== 最終結果 ==========")
    for step, content in result.items():
        print(f"[{step}]：{content}")


if __name__ == "__main__":
    asyncio.run(main())