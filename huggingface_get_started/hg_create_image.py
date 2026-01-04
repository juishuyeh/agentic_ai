import io
import os

import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Hugging Face API 設定
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen-Image"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}


def generate_image(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"width": 1664, "height": 928, "num_inference_steps": 50},
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        # 處理圖片
        image = Image.open(io.BytesIO(response.content))
        return image
    else:
        print(f"錯誤: {response.status_code}, {response.text}")
        return None


# 使用範例
prompt = "A beautiful sunset over mountains, Ultra HD, 4K, cinematic composition."
image = generate_image(prompt)
if image:
    image.save("generated_image.png")