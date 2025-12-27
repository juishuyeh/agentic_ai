from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from prompt_toolkit import prompt
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model("openai:gpt-oss-20b-local", temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# ============ 定義工具 ============
# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


tools = [multiply]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]}, config
    ):
        for value in event.values():
            print("🤖 助手:", value["messages"][-1].content)


while True:
    try:
        user_input = prompt("👤 你: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except Exception:
        # fallback if input() is not available
        user_input = "Hi"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

