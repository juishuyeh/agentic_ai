"""
最小但完整的 ReAct loop —— 用 LangGraph 手刻,不用 prebuilt 黑盒。

架構就三個東西:
  1. State        : 在節點之間流動的資料(這裡就是一串對話訊息)
  2. agent node   : 呼叫 LLM,讓它「想」+ 決定要不要叫工具
  3. tool node    : 真的去執行工具,把結果塞回 State

迴圈長這樣:
        ┌─────────┐
        │  agent  │ ←──────────┐
        └────┬────┘            │
             │ 要叫工具?        │ 工具結果塞回去
        ┌────┴────┐            │
       yes        no       ┌───┴────┐
        │          │        │  tools │
        ▼          ▼        └────────┘
     [tools] ──→ 回 agent     ▲
                              │
        no ──→ END            └── (從 agent 過來)

reason → act → observe → reason → ... 直到模型不再要工具為止。
"""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ─────────────────────────────────────────────────────────────
# 1. 模型設定:走你的 LiteLLM proxy
#    LiteLLM 對外是 OpenAI 相容介面,所以用 ChatOpenAI 接就行。
#    要切本地 / Claude,只改 MODEL 這個字串(LiteLLM 後面去路由)。
# ─────────────────────────────────────────────────────────────
LITELLM_BASE = os.environ.get("LITELLM_BASE", "http://localhost:4000/v1")
LITELLM_KEY = os.environ.get(
    "LITELLM_KEY", "sk-litellm-master-run-key"
)  # LiteLLM 自己的 virtual key

# 第一版建議先用 Claude 把 loop 跑通,確認架構對,再切回本地模型調 prompt。
# MODEL = "claude-sonnet-4"        # 你 LiteLLM 裡 Claude 的 model name
# MODEL = "qwen3-32b"              # 你 omlx 上本地模型的 model name
MODEL = os.environ.get("AGENT_MODEL", "local/gemma-4-26b-a4b")


# ─────────────────────────────────────────────────────────────
# 2. 工具:示範用兩個。@tool 會自動把 docstring + 型別轉成
#    給 LLM 看的 schema,所以 docstring 要寫清楚(模型靠它決定怎麼叫)。
# ─────────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """計算一個數學運算式並回傳結果。
    expression: 合法的 Python 算式字串,例如 "23 * 47 + 12"。
    只能用基本算術,不要傳 import 或函式呼叫。
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return f"錯誤:運算式含有不允許的字元: {expression!r}"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"計算錯誤: {e}"


@tool
def word_count(text: str) -> str:
    """計算一段文字有幾個字(中文按字元、英文按詞)。
    text: 要計算的文字內容。
    """
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    english_words = len([w for w in text.split() if w.isascii() and w.strip()])
    return f"中文字元: {chinese}, 英文詞數: {english_words}"


# create bash
@tool
def execute_bash(command: str) -> str:
    """
    execute a bash command and return its output.
    """
    result = os.popen(command).read()
    return result.strip()


TOOLS = [calculator, word_count, execute_bash]


# ─────────────────────────────────────────────────────────────
# 3. State:LangGraph 的核心觀念。
#    add_messages 是個 reducer——它讓每個節點「附加」訊息而不是覆蓋,
#    所以對話歷史會自動累積。這是整個 loop 能記得前文的關鍵。
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ─────────────────────────────────────────────────────────────
# 4. 綁工具到模型。bind_tools 會把工具 schema 塞進每次 API 呼叫,
#    模型才知道有哪些工具可用、怎麼叫。
# ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=MODEL,
    base_url=LITELLM_BASE,
    api_key=LITELLM_KEY,
    temperature=0,
)
llm_with_tools = llm.bind_tools(TOOLS)


# ─────────────────────────────────────────────────────────────
# 5. 節點定義
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一個會自主規劃的 ReAct agent。
拿到任務時,你先想清楚需要哪些步驟,需要計算或查資料就呼叫對應工具,
拿到工具結果後再決定下一步,直到能給出完整答案為止。
不確定的數字一律用 calculator 算,不要自己心算。"""


def agent_node(state: AgentState) -> dict:
    """reason 階段:讓 LLM 看完整對話歷史,決定下一步(回答 or 叫工具)。"""
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


# tool node 直接用 LangGraph 內建的——它做的事很單純:
# 讀 agent 最後一則訊息裡的 tool_calls,逐一執行,把結果包成 ToolMessage 回傳。
# 這部分沒必要自己刻,刻了也跟內建一樣。
tool_node = ToolNode(TOOLS)


def should_continue(state: AgentState) -> str:
    """條件邊:看 agent 最後一則訊息有沒有要叫工具。
    有 → 去 tools 節點執行;沒有 → 結束,這就是最終答案。
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


# ─────────────────────────────────────────────────────────────
# 6. 組圖:把節點和邊接起來。這就是「ReAct loop」實體化的地方。
# ─────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)

    g.add_edge(START, "agent")  # 進來先 reason
    g.add_conditional_edges("agent", should_continue)  # reason 完決定要不要 act
    g.add_edge("tools", "agent")  # act 完(observe)回去再 reason —— 這條邊就是「迴圈」

    return g.compile()


# ─────────────────────────────────────────────────────────────
# 7. 跑起來。stream 出每一步,你才看得到 loop 在轉。
# ─────────────────────────────────────────────────────────────
def run(task: str):
    graph = build_graph()
    print(f"\n{'=' * 60}\n任務: {task}\n{'=' * 60}")

    for chunk in graph.stream(
        {"messages": [HumanMessage(content=task)]},
        stream_mode="values",
    ):
        msg = chunk["messages"][-1]
        role = type(msg).__name__

        if role == "AIMessage":
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    print(
                        f"\n🤔 [reason→act] 模型決定呼叫工具: {tc['name']}({tc['args']})"
                    )
            elif msg.content:
                print(f"\n✅ [最終答案]\n{msg.content}")
        elif role == "ToolMessage":
            print(f"👁  [observe] 工具回傳: {msg.content}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        # 預設示範:一個需要多步驟 + 工具的任務
        run("幫我算 23*47 再加上 156,然後告訴我『吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮』這句話有幾個中文字")
