import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolNode
from langchain_core.tools import tool

load_dotenv()

@tool
def search_database(query: str, limit: int = 10) -> str:
    """在客戶資料庫中搜尋紀錄。
    Args:
        query: 搜尋關鍵字
        limit: 最大回傳筆數
    """
    return f"Found {limit} results for '{query}'"


tool_node = ToolNode(
    tools=[search_database],
    handle_tool_errors=True
)

# =============== Step 0: 初始化模型 ===============
MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
LLM = init_chat_model(
    MODEL_NAME,
    temperature=0.2,
    max_tokens=2048,
)


agent = create_agent(
    model=LLM,
    tools=tool_node,
    system_prompt="你是一位資料查詢助理。"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "幫我搜尋名字含 Alice 的紀錄"}]
})