
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# ===== 0) 初始化 =====
MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-oss-20b-local")
LLM = init_chat_model(
    MODEL_NAME,
    temperature=0.2,
    max_tokens=2048,
)

# ===== 1) Memory（使用者偏好） =====
MEMORY = {
    "diet": "不吃牛肉",
    "pref": "維也納豬排"
}

# ===== 2) Agent 基底類別 =====
class BaseAgent:
    def __init__(self, name, role_description):
        self.name = name
        self.role_description = role_description

    def act(self, history):
        prompt = f"{self.role_description}\n目前討論紀錄：{history}\n請以 {self.name} 身份回應，簡短發言。"
        resp = str(LLM.invoke(prompt).content)
        return resp.strip()

# ===== 3) 四個角色 =====
class PlannerAgent(BaseAgent):
    def __init__(self):
        role_description = """你是旅行規劃師，提出景點方案。
          輸出 JSON 格式，
          範例：
          {
              "Day1": {"am": "美泉宮", "pm": "聖史蒂芬大教堂"},
              "Day2": {"am": "奧地利國家圖書館", "pm": "維也納歌劇院"}
          }
        """
        super().__init__("Planner", role_description)

class FoodieAgent(BaseAgent):
    def __init__(self):
        role_description = f"""你是美食顧問，根據偏好（{MEMORY['diet']}、偏好 {MEMORY['pref']}）建議餐廳。
          輸出 JSON 格式，
          範例：{{"Day1": "Figlmüller（維也納豬排）", "Day2": "Gasthaus Pöschl"}}
        """
        super().__init__("Foodie", role_description)

class TransportAgent(BaseAgent):
    def __init__(self):
        role_description = """你是交通顧問，自行評估交通方式與所需時間並檢查行程是否交通合理，有無超時風險。
          輸出 JSON 格式，
          範例：
          {
            "Day1": {"am_to_lunch": 25, "lunch_to_pm": 30},
            "Day2": {"am_to_lunch": 22, "lunch_to_pm": 35}
          }
        """
        super().__init__("Transport", role_description)

class ReviewerAgent(BaseAgent):
    def __init__(self):
        role_description = """你是檢視者 (Reviewer)，負責觀察 Planner、Foodie、Transport 的建議。
          你的任務：
          1) 如果有衝突（例如交通超時或餐廳不合適），請條列出來，讓下一回合修正。
          2) 如果沒有衝突，嘗試整合成完整行程 JSON 與文字版方案。
          注意：你不是主管，只是多人對話的一份子，主要任務是檢查和整理。
          3) JSON 格式如下：
          {
              "Day1": {
                  "am": "美泉宮及其花園 (Schloss Schönbrunn)",
                  "lunch": "Figlmüller（維也納豬排）",
                  "pm": "霍夫堡宮 (Hofburg Palace)",
                  "transport": {"am_to_lunch": 39, "lunch_to_pm": 37}
              },
              "Day2": {
                  "am": "聖史蒂芬大教堂 (Stephansdom)",
                  "lunch": "Gasthaus Pöschl",
                  "pm": "維也納藝術史博物館 (Kunsthistorisches Museum Wien)",
                  "transport": {"am_to_lunch": 36, "lunch_to_pm": 38}
              }
          }
        """
        super().__init__("Reviewer", role_description)

# ===== 4) Group Chat 模擬 =====
class GroupChat:
    def __init__(self, agents, rounds=2):
        self.agents = agents
        self.rounds = rounds

    def run(self):
        history = "我們要規劃維也納二日行程，09:00 出發，18:00 前結束。\n"
        for r in range(self.rounds):
            for agent in self.agents:
                reply = agent.act(history)
                history += f"{agent.name}: {reply}\n---\n"
        return history

# ===== 5) 主程式 =====
if __name__ == "__main__":
    agents = [PlannerAgent(), FoodieAgent(), TransportAgent(), ReviewerAgent()]
    chat = GroupChat(agents, rounds=8)
    history = chat.run()
    print("=== Group Chat Demo ===")
    print(history)