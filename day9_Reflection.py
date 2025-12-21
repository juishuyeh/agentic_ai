import json
import os
from datetime import datetime, timedelta

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
# ---- 工具函式：清理 LLM 輸出的 JSON ----
def clean_json_str(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        raw = raw.replace("json", "", 1).strip()
    return raw

# ---- Step 1: 初始行程（模擬 ReAct 輸出，但會超時） ----
init_prompt = """
請幫我規劃今天從上午 9 點出發的行程，包含：
- 上午參觀美泉宮（Schönbrunn Palace）
- 下午參觀聖史蒂芬大教堂（St. Stephen's Cathedral）
- 午餐在市中心的餐廳用餐

條件：
- 每個景點建議 5 小時
- 午餐建議 2 小時
- 景點與餐廳之間交通各 30 分鐘
- 必須在 18:00 前結束

請只輸出 JSON 格式，欄位為 place, minutes。
"""

init_response = str(llm.invoke(init_prompt).content)
raw = clean_json_str(init_response)

try:
    plan = json.loads(raw)
except Exception as e:
    print("初始行程 JSON 解析失敗：", e)
    plan = []

print("=== 初始行程 ===")
print(json.dumps(plan, ensure_ascii=False, indent=2))


# ---- 行程模擬器 ----
def simulate_schedule(plan, title="行程模擬"):
    print(f"\n=== {title} ===")
    time = datetime.strptime("09:00", "%H:%M")
    end = datetime.strptime("18:00", "%H:%M")

    for task in plan:
        duration = task["minutes"]
        finish = time + timedelta(minutes=duration)
        print(f"{time.strftime('%H:%M')}–{finish.strftime('%H:%M')} {task['place']}")
        time = finish

    feasible = time <= end
    print("\n行程可行！" if feasible else "\n行程超時！")
    return feasible


# ---- Step 2: 模擬初始行程 ----
if plan:
    feasible = simulate_schedule(plan, "初始行程模擬")
    if not feasible:
        print("\n初始行程超時，進入 Reflection 檢討...\n")


# ---- Step 3: Reflection 讓 LLM 檢討與改進 ----
refined_plan = []
if plan:
    reflection_prompt = f"""
你剛完成以下一日行程規劃：
{plan}

規劃今天從上午 9 點出發的行程，包含：
  - 上午參觀美泉宮（Schönbrunn Palace）
  - 下午參觀聖史蒂芬大教堂（St. Stephen's Cathedral）
  - 午餐在市中心的餐廳用餐

任務原始條件：
- 每個景點建議 5 小時
- 午餐建議 2 小時
- 景點與餐廳之間交通各 30 分鐘
- 必須在 18:00 前結束

請進行 Reflection：
1. 回顧這份計劃的優點與缺點。
2. 找出可能的問題（例如是否超時，時間分配是否合理）。
3. 提供改進後的版本（JSON 格式，結構與原本相同）。
"""
    reflection_response = str(llm.invoke(reflection_prompt).content)
    print("=== Reflection Output ===")
    print(reflection_response)

    if "[" in reflection_response and "]" in reflection_response:
        try:
            refined_str = reflection_response.split("[", 1)[1]
            refined_str = "[" + refined_str.split("]", 1)[0] + "]"
            refined_str = clean_json_str(refined_str)
            refined_plan = json.loads(refined_str)
        except Exception as e:
            print("\n改進後行程 JSON 解析失敗：", e)


# ---- Step 4: 模擬改進後的行程 ----
if refined_plan:
    simulate_schedule(refined_plan, "改進後行程模擬")
