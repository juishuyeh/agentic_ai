
import json
import os
import re
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from rich.console import Console

# ---- 初始化環境變數與模型（使用 LiteLLM Proxy 的 OpenAI 相容端點） ----
load_dotenv()
MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")

llm = init_chat_model(
    MODEL_NAME,
    temperature=0.3,
    max_tokens=2048,
)

console = Console()


# ---- 模擬工具（維持原始行為） ----
def get_weather(city: str):
    return "今天維也納是下雨天"  # 假設今天下雨


def check_open(place: str):
    closed_places = ["美景宮"]  # 假設美景宮休館
    return f"{place} 正常開放" if place not in closed_places else f"{place} 今日休館"


# ---- 工具路由表（ReAct 解析時使用）----
TOOLS = {
    "查天氣": get_weather,
    "查詢開放狀態": check_open,
}


def clean_json_text(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        # 可能是 ```json\n...\n```
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("\n", 1)[0]
    # 擷取 JSON 陣列部分（Final Answer 可能包含其他說明）
    first = s.find("[")
    last = s.rfind("]")
    if first != -1 and last != -1 and last > first:
        s = s[first : last + 1]
    return s.strip()


# ---- 手寫 ReAct 迴圈（沿用 day7 的 LangChain 呼叫方式） ----
def parse_action_line(response: str):
    """解析經典 ReAct 行格式：Action: 工具(參數)"""
    match = re.search(r"Action:\s*([^\(]+)\((.*?)\)", response, re.S)
    if not match:
        return None
    action, arg = match.groups()
    action = action.strip()
    arg = arg.strip().strip('"').strip("'")
    return action, arg


def parse_tool_metadata_call(response: str):
    """解析類 OpenAI/LiteLLM 的工具呼叫標記：
    例如：<|channel|>commentary to=tool name=查詢開放狀態 <|constrain|>json<|message|>{"place":"美泉\n宮"}
    會回傳 (action_name, argument)。
    """
    meta = re.search(r"to=tool\s+name=([^\s]+).*?<\|message\|>(\{.*?\})", response, re.S)
    if not meta:
        return None
    action = meta.group(1).strip()
    payload_str = meta.group(2).strip()
    try:
        payload = json.loads(payload_str)
    except Exception:
        # 嘗試鬆綁：移除行內換行後再解析
        payload_str_compact = payload_str.replace("\n", "")
        try:
            payload = json.loads(payload_str_compact)
        except Exception:
            return None

    # 根據工具名稱選擇參數欄位
    if action == "查天氣":
        arg = payload.get("city")
    else:
        arg = payload.get("place")

    if isinstance(arg, str):
        arg = arg.replace("\n", "").strip()
    return (action, arg) if arg else None

def react_loop():
    prompt = """你是一個旅行規劃助理，使用 ReAct 模式回答問題，思考請用繁體中文。
格式必須包含以下幾種：
Thought: 你的推理
Action: 你要執行的工具（格式：查天氣(city) 或 查詢開放狀態(place)）
Observation: 工具回傳的結果
Final Answer: 請用 JSON 格式輸出最終行程，例如：
[
  {"place": "美泉宮", "minutes": 120},
  {"place": "聖史蒂芬大教堂", "minutes": 120}
]

約束：每個景點至少 120 分鐘；景點之間交通 30 分鐘；09:00 出發且 18:00 前結束；若休館則調整。

工具僅允許：查天氣(city)、查詢開放狀態(place)。

現在任務：幫我規劃今天去美泉宮、美景宮、聖史蒂芬大教堂的一日行程。"""

    for step in range(8):  # 最多 8 回合
        print(f"\n=== 回合 {step + 1} ===")
        ai_msg = llm.invoke(prompt)
        response = getattr(ai_msg, "content", str(ai_msg))

        # 高亮顯示 Thought / Action / Observation / Final Answer
        parts = re.split(r"(Thought:|Action:|Observation:|Final Answer:)", response)
        if len(parts) == 1:
            console.print(response)
        else:
            if parts[0].strip():
                console.print(parts[0].strip())
            style_map = {
                "Thought:": "bold cyan",
                "Action:": "bold yellow",
                "Observation:": "bold green",
                "Final Answer:": "bold magenta",
            }
            for i in range(1, len(parts), 2):
                token = parts[i]
                content = parts[i + 1] if i + 1 < len(parts) else ""
                style = style_map.get(token, None)
                if style:
                    console.print(token, style=style)
                else:
                    console.print(token)
                if content.strip():
                    console.print(content.strip())

        # ---- 如果是最終答案 ----
        if "Final Answer" in response:
            try:
                json_str = response.split("Final Answer:")[-1].strip()
                json_str = clean_json_text(json_str)
                plan = json.loads(json_str)
                print("\n最終解析後的行程：", plan)
                return plan
            except Exception as e:
                print("\nJSON 解析失敗：", e)
                return []

        # ---- 嘗試解析 Action ----
        parsed = parse_action_line(response)
        if not parsed:
            parsed = parse_tool_metadata_call(response)

        if parsed:
            action, arg = parsed
            if action in TOOLS:
                console.print(f"\n>> 執行工具：{action}({arg})", style="bold yellow")
                try:
                    obs = TOOLS[action](arg)
                except Exception as e:
                    obs = f"工具執行失敗：{e}"
                console.print(f">> 工具回傳：{obs}\n", style="bold green")
                prompt += f"\nObservation: {obs}"
            else:
                console.print(f"\n>> 無效的工具動作：{action}", style="bold red")
                prompt += "\nObservation: 無效的動作"
        else:
            console.print("\n>> 無法解析動作", style="bold red")
            prompt += "\nObservation: 無法解析動作"


# ---- 執行行程模擬 ----
def simulate_schedule(plan, weather="rain"):
    print("\n=== 行程時間模擬 ===")
    time = datetime.strptime("09:00", "%H:%M")
    end = datetime.strptime("18:00", "%H:%M")

    for i, task in enumerate(plan):
        duration = task["minutes"]
        finish = time + timedelta(minutes=duration)
        print(f"{time.strftime('%H:%M')}–{finish.strftime('%H:%M')} 參觀 {task['place']}")
        time = finish

        if i < len(plan) - 1:  # 景點之間交通
            travel_time = int(30 * (1.5 if weather == "rain" else 1))
            finish = time + timedelta(minutes=travel_time)
            if weather == "rain":
                print(f"{time.strftime('%H:%M')}–{finish.strftime('%H:%M')} 前往下一景點（因下雨延誤）")
            else:
                print(f"{time.strftime('%H:%M')}–{finish.strftime('%H:%M')} 前往下一景點")
            time = finish

    print("行程可行！" if time <= end else "行程超時！")


# ---- 主程式 ----
if __name__ == "__main__":
    plan = react_loop()
    if plan:
        simulate_schedule(plan, weather="rain")