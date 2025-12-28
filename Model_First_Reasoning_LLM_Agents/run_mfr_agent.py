from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI  # 外部資訊：非來源內容
from langchain.chat_models import init_chat_model
from rich import print as rprint

load_dotenv()


# 定義狀態，儲存問題、模型以及最終計畫
class AgentState(TypedDict):
    problem_description: str
    problem_model: str
    final_plan: str


# 1. 模型建構階段 (Phase 1: Model Construction)
# 此階段要求 LLM 定義實體、狀態變量、動作（前置條件與影響）及約束 [5], [6]
def model_construction(state: AgentState):
    llm = init_chat_model(model="openai:gpt-oss-20b-local")  # 外部資訊

    prompt = f"""
    分析以下問題。首先，明確定義問題模型，列出：
    (1) 相關實體 (Entities)
    (2) 狀態變量 (State Variables)
    (3) 可能的動作及其前置條件與影響 (Actions with Preconditions and Effects)
    (4) 約束條件 (Constraints)
    
    **在此階段請勿提出解決方案。** [7]
    
    問題：{state["problem_description"]}
    """

    response = llm.invoke(prompt)
    return {"problem_model": response.content}


# 2. 基於模型的推理階段 (Phase 2: Reasoning Over the Model)
# 僅使用第一階段定義的模型來生成步驟計畫，確保遵守約束 [8], [7]
def reasoning_and_planning(state: AgentState):
    llm = init_chat_model(model="openai:gpt-oss-20b-local")  # 外部資訊

    prompt = f"""
    僅使用下方定義的模型，生成一個逐步的解決方案計畫。
    確保所有動作都遵守定義的約束條件和狀態轉換。 [7]
    
    問題模型：
    {state["problem_model"]}
    
    原始問題：
    {state["problem_description"]}
    """

    response = llm.invoke(prompt)
    return {"final_plan": response.content}


# 構建 LangGraph 工作流
workflow = StateGraph(AgentState)

# 新增節點
workflow.add_node("construct_model", model_construction)
workflow.add_node("plan_with_model", reasoning_and_planning)

# 設定邊界：從建構模型開始，完成後進入計畫階段 [3]
workflow.set_entry_point("construct_model")
workflow.add_edge("construct_model", "plan_with_model")
workflow.add_edge("plan_with_model", END)

# 編譯圖形
app = workflow.compile()

# print mermaid diagram
print(app.get_graph().draw_mermaid())

# 執行範例

input_data = {
    "problem_description": "規劃一個多步驟的藥物調度，需考慮不同藥物的交互作用與服用時間。"
}
result = app.invoke(input_data)

print("--- 問題模型 ---")
# print(result['problem_model'])
# 在 terminal 打印出漂亮的格式，使用套件 rich

rprint(result["problem_model"])

print("\n--- 最終計畫 ---")
rprint(result["final_plan"])

