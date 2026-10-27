from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai-proxy.org/v1")
)

try:
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "你好"}]
    )
    print("✅ API 连接成功！回复:", response.choices[0].message.content)
except Exception as e:
    print("❌ API 连接失败:", e)
