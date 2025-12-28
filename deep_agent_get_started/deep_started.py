from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from prompt_toolkit import prompt

load_dotenv()
checkpointer = MemorySaver()
store = InMemoryStore()


def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime), routes={"/memories/": StoreBackend(runtime)}
    )


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."


class WeatherMiddleware(AgentMiddleware):
    tools = [get_weather]


llm = init_chat_model("openai:gpt-oss-20b-local")

agent = create_deep_agent(
    store=store,
    model=llm,
    middleware=[WeatherMiddleware()],
    backend=make_backend,
    checkpointer=checkpointer,
)


def stream_agent_updates(user_input: str):
    config = RunnableConfig(configurable={"thread_id": "1"})
    event = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]}, config=config
    )
    event["messages"][-1].pretty_print()
    print()
    print(
        "================================================================================"
    )


while True:
    try:
        user_input = prompt("User: ")
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            print("Exiting chat.")
            break
        if user_input.strip() == "":
            continue
        stream_agent_updates(user_input)
    except KeyboardInterrupt:
        print("\nExiting chat.")
        break
