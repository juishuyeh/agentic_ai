from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model("openai:gpt-oss-20b-local")
response = model.invoke("為什麼鸚鵡有色彩繽紛的羽毛？")

print(response)