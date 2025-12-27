import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
# ---- 初始化 LLM ----
MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
llm = init_chat_model(
    MODEL_NAME,
    temperature=0.2,
    max_tokens=2048,
)

# ---- 記憶存放結構 ----
memory_store = {"diet": {}, "cuisine_pref": []}

def update_memory(user_input: str):
    if "不吃牛肉" in user_input:
        memory_store["diet"]["beef"] = False
    if "豬排" in user_input and "schnitzel" not in memory_store["cuisine_pref"]:
        memory_store["cuisine_pref"].append("schnitzel")

# 使用者輸入
user_input = "我不吃牛肉，但喜歡維也納豬排"
update_memory(user_input)

print("使用者記憶：", memory_store)
candidates = [
    {"name": "市中心牛排館", "tags": ["beef", "steak"]},
    {"name": "市中心壽司店", "tags": ["japanese", "sushi"]},
    {"name": "市中心維也納豬排餐廳", "tags": ["schnitzel", "austrian"]}
]


prompt = f"""
使用者的飲食習慣：
{memory_store}

候選餐廳：
{candidates}

請依照使用者的飲食限制與偏好，選擇最合適的一間餐廳，
並只輸出餐廳的名稱。
"""

response = str(llm.invoke(prompt).content)
print("AI 建議的餐廳：", response.strip())