import os

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()

# =============== Step 0: 初始化模型 ===============
MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
LLM = init_chat_model( MODEL_NAME, temperature=0, max_tokens=2048)

# =============== Step 1: 定義工具 ===============

API_KEY = os.getenv("ACCUWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("請先設定 ACCUWEATHER_API_KEY 環境變數")

# --------- 定義工具 ---------
@tool
def accuweather_search_city(city: str, country_code: str = "AT") -> dict:
    """用 AccuWeather 搜尋城市並回傳 Location Key。"""
    url = "https://dataservice.accuweather.com/locations/v1/cities/search"
    params = {"apikey": API_KEY, "q": city, "language": "zh-tw"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        return {"error": f"找不到城市：{city}"}
    return data[0]

@tool
def accuweather_current_conditions(location_key: str) -> dict:
    """查詢指定 Location Key 的即時天氣。"""
    url = f"https://dataservice.accuweather.com/currentconditions/v1/{location_key}"
    params = {"apikey": API_KEY, "language": "zh-tw", "details": "true"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data[0] if data else {"error": f"查不到天氣：{location_key}"}

tools = [accuweather_search_city, accuweather_current_conditions]

# --------- 初始化模型 ---------
agent = create_agent(
    model=LLM,
    tools=tools,
    system_prompt="你是專業氣象助理。查詢天氣時，先用 accuweather_search_city (city 傳入中文即可) 找 Location Key，再用 accuweather_current_conditions。"
)

# --------- 執行 ---------
resp = agent.invoke({
    "messages": [{"role": "user", "content": "幫我查維也納的即時天氣"}]
})

print("\n=== 對話歷程 ===")
for m in resp["messages"]:
    role = m.__class__.__name__
    print(f"[{role}] {getattr(m, 'content', '') or getattr(m, 'tool_calls', '')}")

print("\n=== 模型最終回答 ===")
print(resp["messages"][-1].content)