# langgraph_use_mcp_as_server.py
from datetime import datetime
from typing import List, TypedDict

from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

load_dotenv()
# 定義狀態類型
class TextProcessState(TypedDict):
    input_text: str
    processed_text: str
    ai_response: str
    steps: List[str]

# 初始化模型
model = init_chat_model(
    model="openai:openai/gpt-oss-20b-local",
    temperature=0.7
)

# 建立 FastMCP 實例
mcp = FastMCP("Simple-FastMCP-LangGraph")

def create_text_processing_graph():
    """建立文字處理的 LangGraph 工作流"""
    
    async def preprocess_text(state: TextProcessState) -> TextProcessState:
        """預處理文字"""
        processed = state["input_text"].strip()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            **state,
            "processed_text": processed,
            "steps": state["steps"] + [f"文字預處理完成 ({timestamp})"]
        }
    
    async def generate_ai_response(state: TextProcessState) -> TextProcessState:
        """生成 AI 回應"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一個專業的文字分析助手。"),
            ("human", "請分析以下文字：\n\n{text}")
        ])
        
        response = await model.ainvoke(
            prompt.format_messages(text=state["processed_text"])
        )
        
        return {
            **state,
            "ai_response": response.content,
            "steps": state["steps"] + ["AI 分析完成"]
        }
    
    # 建立工作流圖
    workflow = StateGraph(TextProcessState)
    workflow.add_node("preprocess", preprocess_text)
    workflow.add_node("ai_analyze", generate_ai_response)
    
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "ai_analyze")
    workflow.add_edge("ai_analyze", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# 建立全域 LangGraph 實例
text_processor = create_text_processing_graph()

# 註冊為 MCP 工具
@mcp.tool()
async def process_text_with_langgraph(text: str, ctx: Context = None) -> str:
    """
    使用 LangGraph 處理文字
    
    Args:
        text: 要處理的文字內容
    
    Returns:
        處理結果
    """
    if ctx:
        await ctx.info(f"開始分析文字: {text[:30]}...")
    
    # 初始狀態
    initial_state = {
        "input_text": text,
        "processed_text": "",
        "ai_response": "",
        "steps": []
    }
    
    # 執行工作流
    final_state = await text_processor.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": f"analyze_{datetime.now()}"}}
    )
    
    # 格式化結果
    result = f"""📊 文字分析結果

📝 原始文字:
{final_state['input_text']}

🤖 AI 分析:
{final_state['ai_response']}

⚙️ 處理步驟:
{' → '.join(final_state['steps'])}"""
    
    return result

if __name__ == "__main__":
    print("🚀 啟動 Simple FastMCP + LangGraph 伺服器")
    print("🌐 伺服器地址: http://127.0.0.1:8004/mcp")
    
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8004,
        log_level="info"
    )