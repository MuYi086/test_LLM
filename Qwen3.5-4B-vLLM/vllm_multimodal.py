# vllm_multimodal.py
from openai import OpenAI

client = OpenAI(api_key="sk-xxx", base_url="http://127.0.0.1:8391/v1")

response = client.chat.completions.create(
    model="Qwen3.5-4B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://mms-graph.cdn.bcebos.com/activity/pc/shitu_b01.jpg"}},
            {"type": "text", "text": "请描述这张图片的内容。"},
        ],
    }],
)
print(response.choices[0].message.content)