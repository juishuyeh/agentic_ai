import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from prompt_toolkit import prompt
from typing_extensions import TypedDict

# 載入環境變數
load_dotenv()

# ============ 1. 設定 MCP 連線參數 ============
server_params = {
    "url": "http://localhost:4000/mcp/dwhttp",
    "headers": {
        "Authorization": f"Bearer {os.getenv('LITELLM_API_KEY', 'mykey')}"
    }
}

# ============ 2. 定義 Graph State ============
class State(TypedDict):
    messages: Annotated[list, add_messages]

async def main():
    print(f"🔌 正在連線至 MCP Server: {server_params['url']} ...")

    # ============ 3. 建立並保持 MCP 連線 ============
    # 注意：LangGraph 的生命週期必須在這個 async with 區塊內
    async with streamablehttp_client(**server_params) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 動態載入 MCP 工具
            tools = await load_mcp_tools(session)
            print(f"✅ 成功載入 {len(tools)} 個 MCP 工具: {[t.name for t in tools]}")

            # ============ 4. 建構 LangGraph ============
            # 初始化模型 (請確保此模型支援 Tool Calling)
            llm = init_chat_model("openai:openai/gpt-oss-20b-local", temperature=0)
            
            # 綁定工具
            llm_with_tools = llm.bind_tools(tools)

            # 定義 Chatbot 節點函數
            def chatbot(state: State):
                return {"messages": [llm_with_tools.invoke(state["messages"])]}

            # 建立圖表
            graph_builder = StateGraph(State)
            graph_builder.add_node("chatbot", chatbot)
            
            # 使用 LangGraph 內建的 ToolNode 來執行 MCP 工具
            tool_node = ToolNode(tools=tools)
            graph_builder.add_node("tools", tool_node)

            # 設定邊與條件
            graph_builder.add_conditional_edges(
                "chatbot",
                tools_condition,
            )
            graph_builder.add_edge("tools", "chatbot")
            graph_builder.add_edge(START, "chatbot")

            # 加入記憶體
            memory = InMemorySaver()
            graph = graph_builder.compile(checkpointer=memory)

            # ============ 5. 進入對話迴圈 ============
            print("\n🤖 系統就緒！可以開始對話 (輸入 'q' 離開)")
            
            # 設定 thread_id 以啟用記憶功能
            config = {"configurable": {"thread_id": "1"}}

            while True:
                try:
                    # 為了不阻塞 asyncio loop，這裡用 run_in_executor 或者直接用 prompt
                    # 在簡單的 CLI 應用中，直接呼叫 prompt() 通常是可以接受的
                    user_input = await asyncio.to_thread(prompt, "👤 你: ")
                    user_input = user_input.strip()

                    if user_input.lower() in ["quit", "exit", "q"]:
                        print("Goodbye!")
                        break

                    # 串流執行 Graph
                    print("🤖 助手: ", end="", flush=True)
                    async for event in graph.astream(
                        {"messages": [{"role": "user", "content": user_input}]}, 
                        config,
                        stream_mode="updates" # 只關注更新的部分
                    ):
                        for node_name, value in event.items():
                            # 我們只印出 chatbot 產生的最後一條訊息內容
                            if node_name == "chatbot":
                                last_msg = value["messages"][-1]
                                if last_msg.content:
                                    print(last_msg.content)
                                # 如果是 Tool call，通常不會有 content，所以不用印
                            elif node_name == "tools":
                                # 可以選擇是否印出工具執行的結果，這裡選擇略過保持畫面乾淨
                                pass

                    print("") # 換行準備下一次輸入

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ 發生錯誤: {e}")
                    break

if __name__ == "__main__":
    asyncio.run(main())