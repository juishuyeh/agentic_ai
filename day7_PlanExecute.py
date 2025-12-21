import json
import os
from datetime import datetime, timedelta
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# ---- 初始化環境變數 ----
load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")

llm = init_chat_model(
	MODEL_NAME,
	temperature=0.2,
	max_tokens=2048,
)

def clean_json_text(text: str) -> str:
	"""移除可能的程式碼圍欄與多餘內容，保留純 JSON 字串。"""
	s = text.strip()
	# 去除 ```json ... ``` 或 ``` ... ```
	if s.startswith("```"):
		# 可能是 ```json\n...\n```
		s = s.split("\n", 1)[1] if "\n" in s else s
		if s.endswith("```"):
			s = s.rsplit("\n", 1)[0]
	# 嘗試截取第一個 [ 到最後一個 ]
	first = s.find("[")
	last = s.rfind("]")
	if first != -1 and last != -1 and last > first:
		s = s[first : last + 1]
	return s.strip()

def request_itinerary() -> dict[str, Any]:
	"""向模型請求行程並解析為 JSON。"""
	query = (
		"請規劃今天從上午 9 點出發，參觀維也納三個景點：\n"
		"美泉宮、美景宮、聖史蒂芬大教堂。\n"
		"條件：\n"
		"- 每個景點至少 2 小時\n"
		"- 景點之間交通 30 分鐘\n"
		"- 必須在 18:00 前結束\n"
		"請只輸出 JSON 格式，每個步驟包含：\n"
		"title, place, minutes, indoor (是否室內, true/false)。"
	)

	# 直接用單訊息呼叫（LangChain 會自動包裝為 HumanMessage）
	ai_msg = llm.invoke(query)
	raw = getattr(ai_msg, "content", str(ai_msg))
	print("AI 原始輸出：", raw)

	cleaned = clean_json_text(raw)
	try:
		data = json.loads(cleaned)
		return data
	except Exception:
		# 若無法解析為物件，可能是陣列或其他結構，再嘗試更寬鬆解析
		try:
			# 有些模型會輸出陣列；直接嘗試解析
			return json.loads(cleaned)
		except Exception:
			# 最後退回原始字串
			return {"raw": raw}

# Step 3: Tool (模擬天氣 API)
def get_weather(city: str):
    return "rain"  # 假設今天下雨

data = request_itinerary()
print("解析後 JSON：", json.dumps(data, ensure_ascii=False, indent=2))

# Step 4: 執行計劃
weather = get_weather("Vienna")
print("\n今日天氣：", weather)


time = datetime.strptime("09:00", "%H:%M")
end = datetime.strptime("18:00", "%H:%M")


for task in data:
    duration = task["minutes"]
    if weather == "rain" and "前往" in task["title"]:
        duration = int(duration * 1.5)  # 下雨天交通延長 1.5 倍
        task["title"] += "（因下雨延誤）"

    finish = time + timedelta(minutes=duration)
    print(f"{time.strftime('%H:%M')}–{finish.strftime('%H:%M')} {task['title']}")
    time = finish
    print("行程可行！" if time <= end else "行程超時！")
