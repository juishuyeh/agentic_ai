import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

# ===== 0) 初始化 =====
load_dotenv()

# 1. 定義工具（用 @tool 包裝）
@tool
def get_weather(city: str) -> str:
    """查詢指定城市的天氣狀況。"""
    weather_data = {
        "維也納": "大晴天，氣溫 22°C",
        "Vienna": "Sunny, 22°C",
        "台北": "多雲，氣溫 30°C",
    }
    return weather_data.get(city, "查無資料")

# 2. 初始化 Gemini 模型
MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
LLM = init_chat_model(
    MODEL_NAME,
    temperature=0.2,
    max_tokens=2048,
)
# 3. 建立 Agent（支援 tool calling）
agent = create_agent(
    model=LLM,
    tools=[get_weather],
)

# 4. 使用者查詢
query = "請告訴我今天維也納的天氣"

# 5. 由 Agent 決定是否呼叫工具
result = agent.invoke({"messages": [{"role": "user", "content": query}]})

# 6. 輸出最終回覆
print(result["messages"][-1].content)