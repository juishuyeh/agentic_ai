# langgraph_use_mcp_as_client.py
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

# 初始化模型
model = init_chat_model( "openai:openai/gpt-oss-20b-local", temperature=0)

# 設定 MCP 客戶端
client = MultiServerMCPClient({
    "math": {
        "url": "http://127.0.0.1:8000/mcp/",
        "transport": "http",
    }
})

async def main():
    # 從 MCP 伺服器取得工具
    tools = await client.get_tools()
    
    # 綁定工具到模型
    model_with_tools = model.bind_tools(tools)
    
    # 建立工具節點
    tool_node = ToolNode(tools)
    
    # 定義條件判斷
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    # 定義模型調用函數
    async def call_model(state: MessagesState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
    # 建立圖
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue)
    builder.add_edge("tools", "call_model")
    
    # 編譯並執行
    graph = builder.compile()
    
    # 測試
    result = await graph.ainvoke({
        "messages": [{"role": "user", "content": "what's (3 + 5) x 12 + 8 + 8 + 12?, Use tools. and step-by-step reasoning."}]
    })

    print("Final Response:", result["messages"][-1].content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())