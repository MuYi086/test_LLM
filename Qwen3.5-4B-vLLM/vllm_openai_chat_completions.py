# vllm_openai_chat_completions.py
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxx",
    base_url="http://127.0.0.1:8391/v1",
)

# 非思考模式：传入 enable_thinking=false
chat_outputs = client.chat.completions.create(
    model="Qwen3.5-4B",
    messages=[{"role": "user", "content": "用一句话介绍深度学习。"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(chat_outputs.choices[0].message.content)