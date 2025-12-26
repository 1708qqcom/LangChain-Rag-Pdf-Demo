from openai import OpenAI

client = OpenAI(
    api_key = 'sk-xxxxx',
    base_url = 'https://api.deepseek.com/v1'
)

response = client.chat.completions.create(
    model = 'deepseek-chat',
    messages = [
        {'role': 'system', 'content': '你是一个超级幽默、专业的助手'},
        {'role': 'user', 'content': '给我讲个大学生专属冷笑话'}
    ],
    temperature = 0.8
)

print(response.choices[0].message.content)