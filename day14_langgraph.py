import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph

load_dotenv()

# ===== 0) 初始化 =====
MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
LLM = init_chat_model(
    MODEL_NAME,
    temperature=0.2,
    max_tokens=2048,
)

# 初始化 Gemini

# 定義狀態
class State(TypedDict):
    message: str

# 節點
def start_node(state: State):
    resp = LLM.invoke(f"Say hello to {state['message']}")
    return {"message": resp.content}

def end_node(state: State):
    print("LangGraph Output:", state["message"])
    return state

# 建立圖
graph = StateGraph(State)
graph.add_node("start", start_node)
graph.add_node("end", end_node)

graph.set_entry_point("start")
graph.add_edge("start", "end")

app = graph.compile()
res = app.invoke({"message": "World"})
print("Final Result:", res)
