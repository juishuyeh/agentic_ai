import json
import os
import random
import re
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

def safe_json_loads(text, fallback=None):
    """
    嘗試解析 JSON，會自動移除 ```json ... ``` code fence。
    如果失敗，就回傳 fallback。
    """
    if not text:
        return fallback
    # 移除 Markdown code block 標記
    clean = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE|re.MULTILINE)
    try:
        return json.loads(clean)
    except Exception:
        return fallback

# ===== 0) 初始化 =====
MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
LLM = init_chat_model(
    MODEL_NAME,
    temperature=0.2,
    max_tokens=2048,
)

# ===== 1) Memory (使用者偏好) =====
MEMORY = {
    "diet": "不吃牛肉",
    "pref": "維也納豬排"
}

# ===== 2) Tool Use 模擬 =====
def travel_time_tool(from_place, to_place):
    """
    模擬呼叫外部 API 查交通時間。
    這裡用隨機數 20–40 分鐘。
    """
    return random.randint(20, 40)

# ===== 3) Agents 定義 =====
def planner_agent():
    prompt = """
    請規劃維也納 2 日行程，每天上午與下午各安排一個景點，輸出 JSON 格式。
    範例：
    {
        "Day1": {"am": "美泉宮", "pm": "聖史蒂芬大教堂"},
        "Day2": {"am": "奧地利國家圖書館", "pm": "維也納歌劇院"}
    }
    """
    resp = str(LLM.invoke(prompt).content)
    try:
        return safe_json_loads(resp)
    except Exception:
        return None

def foodie_agent(plan):
    prompt = f"""根據以下行程，挑選午餐餐廳。
需符合條件：{MEMORY['diet']}，並偏好 {MEMORY['pref']}。
輸出 JSON 格式，
範例：{{"Day1": "Figlmüller（維也納豬排）", "Day2": "Gasthaus Pöschl"}}
每天一間餐廳。行程: {plan}"""
    resp = str(LLM.invoke(prompt).content)
    try:
        return safe_json_loads(resp)
    except Exception:
        return None

def transport_agent(plan, food):
    """
    Transport Agent：呼叫 travel_time_tool 取得所有需要的時間，
    再交給 LLM 整理成 {DayX: {am_to_lunch, lunch_to_pm}} 的格式。
    """
    # Step 1: 先收集原始資料
    raw_data = []
    for day, schedule in plan.items():
        am_to_lunch = travel_time_tool(schedule["am"], food[day])
        lunch_to_pm = travel_time_tool(food[day], schedule["pm"])
        raw_data.append({
            "day": day,
            "am": schedule["am"],
            "lunch": food[day],
            "pm": schedule["pm"],
            "am_to_lunch": am_to_lunch,
            "lunch_to_pm": lunch_to_pm
        })

    # Step 2: 請 LLM 整理成結構化 JSON
    prompt = f"""
    以下是交通估算的原始資料，請整理成 JSON 格式：
    {raw_data}

    格式範例：
    {{
      "Day1": {{"am_to_lunch": 25, "lunch_to_pm": 30}},
      "Day2": {{"am_to_lunch": 22, "lunch_to_pm": 35}}
    }}
    """
    resp = str(LLM.invoke(prompt).content)
    return safe_json_loads(resp, {})

def coordinator_agent(plan, food, transit):
    """
    Coordinator Agent：統籌行程，檢查是否超時 (09:00 ~ 18:00)
    """
    day_start = datetime.strptime("09:00", "%H:%M")
    end_limit = datetime.strptime("18:00", "%H:%M")

    result = {}
    for day, schedule in plan.items():
        cur = day_start
        steps = []

        # 上午景點
        steps.append(f"{cur.strftime('%H:%M')}–{(cur:=cur+timedelta(minutes=180)).strftime('%H:%M')} {schedule['am']}")

        # 上午 → 午餐
        cur += timedelta(minutes=transit[day]["am_to_lunch"])
        steps.append(f"{cur.strftime('%H:%M')} 午餐：{food[day]} (90 分鐘)")
        cur += timedelta(minutes=90)

        # 午餐 → 下午
        cur += timedelta(minutes=transit[day]["lunch_to_pm"])
        steps.append(f"{cur.strftime('%H:%M')}–{(cur:=cur+timedelta(minutes=150)).strftime('%H:%M')} {schedule['pm']}")

        feasible = "行程可行！" if cur <= end_limit else "行程超時！"
        result[day] = {"timeline": steps, "status": feasible}

    return result

# ===== 4) 執行 Demo =====
if __name__ == "__main__":
    print("=== Multi-Agent + Tool Use Demo ===")

    plan = planner_agent()
    print("Planner:", plan)

    food = foodie_agent(plan)
    print("Foodie:", food)

    transit = transport_agent(plan, food)
    print("Transport (Tool Use):", transit)

    final = coordinator_agent(plan, food, transit)
    print("\n=== 最終統籌結果 ===")
    for day, info in final.items():
        print(f"\n{day}")
        for s in info["timeline"]:
            print(" ", s)
        print(" ", info["status"])
