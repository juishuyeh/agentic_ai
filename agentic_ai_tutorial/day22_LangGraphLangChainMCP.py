import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, MessagesState, StateGraph

load_dotenv()

class State(MessagesState):
    plan: str = ""
    check: str = ""
    adjusted: str = ""
    output: str = ""

# Step 1: 初始化 MCP 工具
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
    return client, tools


# Step 2: 建立 LangChain Agent
async def build_agent(tools):
    MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
    LLM = init_chat_model(MODEL_NAME, temperature=0.2)
    agent = create_agent(
        model=LLM,
        tools=tools,
        system_prompt=(
            "你是一位智慧旅遊助理，根據天氣與景點資料，"
            "推薦最合適的維也納一日行程。必要時可查天氣、查景點或檢查營業狀況。"
        )
    )
    return agent


# Step 3: 定義 LangGraph 流程節點
async def main():
    client, tools = await init_tools()
    agent = await build_agent(tools)

    graph = StateGraph(State)

    async def step_plan(state: State):
        print("\n[PLAN] 模型規劃中...")
        user_msg = HumanMessage(content="請幫我規劃維也納今天的行程")
        convo = (state.get("messages") or []) + [user_msg]
        result = await agent.ainvoke({"messages": convo})
        ai_msg = result["messages"][-1]
        return {"messages": [ai_msg], "plan": ai_msg.content}

    async def step_check(state: State):
        print("\n[CHECK] 模型確認是否需查營業狀況...")
        plan_text = state.get("plan", "")
        prompt = f"請確認以下行程中的景點是否開放，並清楚列出休館者：\n{plan_text}"
        user_msg = HumanMessage(content=prompt)
        convo = (state.get("messages") or []) + [user_msg]
        result = await agent.ainvoke({"messages": convo})
        ai_msg = result["messages"][-1]
        return {"messages": [ai_msg], "check": ai_msg.content}

    async def step_adjust(state: State):
        print("\n[ADJUST] 若有休館，重新規劃...")
        plan_text = state.get("plan", "")
        check_text = state.get("check", "")
        prompt = (
            "若有景點關閉，請以相近主題或地理位置的替代景點重新安排行程，並重排動線：\n"
            f"原行程：\n{plan_text}\n\n休館項目：\n{check_text}"
        )
        user_msg = HumanMessage(content=prompt)
        convo = (state.get("messages") or []) + [user_msg]
        result = await agent.ainvoke({"messages": convo})
        ai_msg = result["messages"][-1]
        return {"messages": [ai_msg], "adjusted": ai_msg.content}

    async def summarize(state: State):
        print("\n[SUMMARY] 最終結果整合...")
        user_msg = HumanMessage(content="請基於目前所有對話與決策，整理最終的一日行程與時程表")
        convo = (state.get("messages") or []) + [user_msg]
        result = await agent.ainvoke({"messages": convo})
        ai_msg = result["messages"][-1]
        print("\nDone.")
        print(ai_msg.content)
        return {"messages": [ai_msg], "output": ai_msg.content}

    # LangGraph 節點設計
    graph.add_node("plan", step_plan)
    graph.add_node("check", step_check)
    graph.add_node("adjust", step_adjust)
    graph.add_node("summary", summarize)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "check")

    # 條件分支：僅在確定有休館時，進入替代規劃
    def needs_adjust(state: State):
        text = (state.get("check") or "").lower()
        return "adjust" if ("休館" in text or "closed" in text or "not open" in text) else "summary"

    graph.add_conditional_edges("check", needs_adjust, {
        "adjust": "adjust",
        "summary": "summary",
    })

    graph.add_edge("adjust", "summary")
    graph.add_edge("summary", END)

    app = graph.compile()
    result = await app.ainvoke({})
    print("\n最終輸出：", result["output"])



if __name__ == "__main__":
    asyncio.run(main())
