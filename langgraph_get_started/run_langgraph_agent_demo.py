import operator
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import (AnyMessage, HumanMessage, SystemMessage,
                                ToolMessage)
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

load_dotenv()

# 初始化聊天模型
model = init_chat_model("openai:gpt-oss-20b-local", temperature=0)

# ============ 定義工具 ============
@tool
def multiply(a: int, b: int) -> int:
    """將 `a` 和 `b` 相乘"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """將 `a` 和 `b` 相加"""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """將 `a` 除以 `b`"""
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# ============ 定義狀態結構 ============
class MessagesState(TypedDict):
    """代理的狀態結構：包含訊息歷史和LLM呼叫次數"""
    messages: Annotated[list[AnyMessage], operator.add]  # 自動合併訊息列表
    llm_calls: int

# ============ 定義節點（Node）============

def llm_call(state: MessagesState) -> dict:
    """
    LLM節點：決定是否呼叫工具
    - 輸入：當前狀態（包含訊息歷史）
    - 輸出：LLM的回應訊息和呼叫計數
    """
    response = model_with_tools.invoke(
        [SystemMessage(content="你是一個有用的助手，用於執行算術運算。")]
        + state["messages"]
    )
    
    return {
        "messages": [response],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: MessagesState) -> dict:
    """
    工具執行節點：執行LLM要求的工具呼叫
    - 提取最後一條訊息中的tool_calls
    - 執行對應的工具
    - 返回工具執行結果作為ToolMessage
    """
    result = []
    last_message = state["messages"][-1]
    
    # 檢查AIMessage是否有tool_calls屬性
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
    
    return {"messages": result}

# ============ 定義條件邊（Conditional Edge）============

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """
    路由邏輯：決定是否繼續執行或結束
    - 如果LLM生成tool_calls → 路由到tool_node
    - 否則 → 結束流程
    """
    last_message = state["messages"][-1]
    
    # 檢查是否有tool_calls（代表需要執行工具）
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    
    return END

# ============ 構建代理圖 ============

agent_builder = StateGraph(MessagesState)

# 添加節點
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# 定義邊（流程連接）
agent_builder.add_edge(START, "llm_call")  # 開始 → LLM節點
agent_builder.add_conditional_edges(
    "llm_call",  # 從 "llm_call" 節點出發
    should_continue,  # 使用 should_continue 函數決定下一步
    {"tool_node": "tool_node", END: END}  # 條件路由映射：
                                          # - 如果 should_continue 返回 "tool_node"，則轉到 "tool_node" 節點
                                          # - 如果 should_continue 返回 END，則結束流程
)
agent_builder.add_edge("tool_node", "llm_call")  # 工具執行 → 回到LLM

# 編譯代理
agent = agent_builder.compile()

# ============ 執行代理 ============

messages = [HumanMessage(content="計算 3 加 4, 計算 10 乘以 2, 然後將結果相加")]
result = agent.invoke({"messages": messages, "llm_calls": 0})

print("\n=== 執行結果 ===")
for m in result["messages"]:
    m.pretty_print()

mermaid_code = agent.get_graph().draw_mermaid()
print("\n=== Mermaid 圖示 ===")
print(mermaid_code)