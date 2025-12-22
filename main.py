"""
LangChain + LiteLLM Agent 最終版
包含清晰的回應提取和顯示
"""
import json
import os
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

# ===== 配置區 =====
LITELLM_BASE_URL = "http://localhost:4000"
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "your_litellm_api_key")
MODEL_NAME = "gemma-3-12b"

# ===== 系統提示詞 =====
SYSTEM_PROMPT = """你是一位專業的天氣助手。

你可以使用兩個工具:
- get_weather_for_location: 用於獲取特定城市的天氣
- get_user_location: 用於獲取使用者的位置

請用繁體中文回覆，態度友善專業。

IMPORTANT: 只回答天氣相關的問題。如果使用者問你其他問題，請禮貌地告訴他們你只能回答天氣相關的問題。

"""

# ===== 上下文結構 =====
@dataclass
class Context:
    """執行時期上下文"""
    user_id: str

# ===== 工具定義 =====
@tool
def get_weather_for_location(city: str) -> str:
    """獲取指定城市的天氣。"""
    # 這裡可以串接真實的天氣 API
    weather_data = {
        "台北": "晴天，25°C，濕度 60%",
        "台南": "多雲，28°C，濕度 70%",
        "高雄": "晴天，30°C，濕度 65%",
    }
    return weather_data.get(city, f"{city} 天氣晴朗，溫度適中")

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根據使用者 ID 檢索使用者位置。"""
    # 這裡可以從資料庫查詢使用者位置
    user_locations = {
        "1": "台北",
        "2": "台南",
        "3": "高雄",
    }
    user_id = runtime.context.user_id
    return user_locations.get(user_id, "台北")

# ===== 回應格式 =====
@dataclass
class WeatherResponse:
    """天氣回應格式"""
    answer: str  # 主要回答
    weather_info: str | None = None  # 天氣資訊（如果有的話）


model = ChatOpenAI(
    base_url=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    model=MODEL_NAME,
    temperature=0.7,
)

# ===== 建立 Agent =====
print("🤖 建立 Agent...")
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(WeatherResponse),
    checkpointer=checkpointer
)

# ===== 輔助函數 =====
def extract_response(response: dict) -> str:
    """從 Agent 回應中提取最終答案"""
    if 'messages' not in response:
        return str(response)
    
    # 取得最後一條 AI 訊息
    messages = response['messages']
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.content:
            content = msg.content.strip()
            # 如果是結構化回應格式
            if content.startswith('[ResponseFormat]'):
                try:
                    # 提取 JSON 部分
                    json_str = content.split('[ResponseFormat]\n')[1].split('\n[END_ResponseFormat]')[0]
                    data = json.loads(json_str)
                    return data.get('answer', content)
                except Exception:
                    pass
            # 一般回應
            if content and not content.startswith('[') and content != '':
                return content
    
    return "抱歉，無法生成回應"

def chat(user_message: str, config: dict, context: Context) -> str:
    """執行對話並返回回應"""
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        context=context
    )
    return extract_response(response)

# ===== 主程式 =====
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌤️  LangChain + LiteLLM 天氣助手")
    print("="*60 + "\n")
    
    # 對話設定
    user_id = "1"  # 使用者 ID
    thread_id = "weather-chat-001"  # 對話 ID
    config = {"configurable": {"thread_id": thread_id}}
    context = Context(user_id=user_id)
    
    # 互動模式
    print("\n💬 進入互動模式（輸入 'exit' 結束）\n")
    
    while True:
        try:
            user_input = input("👤 你: ").strip()
            if user_input.lower() in ['exit', 'quit', '結束', '離開']:
                print("\n👋 再見！")
                break
            
            if not user_input:
                continue
            
            response_text = chat(user_input, config, context)
            print(f"🤖 助手: {response_text}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 再見！")
            break
        except Exception as e:
            print(f"❌ 錯誤: {e}\n")