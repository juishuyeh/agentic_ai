import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (BaseMessage, HumanMessage, SystemMessage,
                                     ToolCall)
from langchain_core.tools import tool
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

load_dotenv()

# =============== Step 0: 初始化模型 ===============
MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
LLM = init_chat_model(
    MODEL_NAME,
    temperature=0.2,
    max_tokens=2048,
)

# =============== Step 1: 定義工具：查天氣、查距離（順路） ===============
@tool
def get_weather(location: str, period: str) -> str:
    """查詢指定地點在某時段的天氣摘要"""
    if "維也納" in location and "下午" in period:
        return "維也納下午：局部陣雨，建議改室內。"
    return "晴時多雲"

@tool
def check_distance(start: str, end: str) -> str:
    """查詢兩個景點是否順路"""
    if start == "納許市場" and end == "藝術史博物館":
        return "步行 15 分鐘，路線合理。"
    if start == "納許市場" and end == "美泉宮花園":
        return "電車約 12 分鐘，路線合理。"
    return "需轉乘，不太順路。"

tools = [get_weather, check_distance]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = LLM.bind_tools(tools)

# =============== Step 2: One-shot CoT 提示 ===============
SYSTEM_PROMPT = """
你是一位維也納旅行助理，請以「逐步思考 → 查詢資訊 → 最終結論」的格式回答。
"""

# =============== Step 3: LLM 節點（推理） ===============
@task
def call_llm(messages: list[BaseMessage]):
    resp = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    print("\n[LLM 推理輸出]")
    print(resp.pretty_repr())  # 顯示推理步驟與可能的 tool call
    return resp

# =============== Step 4: 工具節點（執行查詢） ===============
@task
def call_tool(tool_call: ToolCall):
    tool = tools_by_name[tool_call["name"]]
    print(f"\n[執行查詢] {tool_call['name']} with args {tool_call['args']}")
    result = tool.invoke(tool_call)
    print(f"[查詢結果] {result.content}")
    return result

# =============== Step 5: Orchestrator（一次推理即可） ===============
@entrypoint()
def cot_agent(messages: list[BaseMessage]):
    llm_response = call_llm(messages).result()
    while llm_response.tool_calls:
        tool_results = [call_tool(tc).result() for tc in llm_response.tool_calls]
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = call_llm(messages).result()
    return add_messages(messages, llm_response)

# =============== 測試 ===============
question = (
    "早上參觀聖史蒂芬大教堂，中午在納許市場用餐；"
    "請依下午天氣決定：若下雨改去藝術史博物館，否則去美泉宮花園。"
    "並確認從納許市場出發是否順路，請用 Step 說明。"
)

final_messages = cot_agent.invoke([HumanMessage(content=question)])

print("\n[最終答案]")
print(final_messages[-1].content)