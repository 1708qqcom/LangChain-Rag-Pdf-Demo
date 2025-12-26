import os
from openai import OpenAI

os.environ["DEEPSEEK_API_KEY"] = "sk-xxxxx"

client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com/v1"
)

history = [ # system就是对模型设定基础人设或规则（只在最开始定义一次）
    {"role": "system", "content": "你现在是一个搞怪但幽默的大学生，专门吐槽学习人工智能的困难。"}
]

# === 可选：先做一次固定需求的吐槽（并加入历史）===
initial_demand = "为按钮点击事件添加可配置的动态效果，"

print("【需求】")
print(initial_demand)

# 构造第一次 user 消息（带完整指令）,下面举例是产品经理和程序员的幽默对话
first_user_msg = (
    "请帮我把下面这个产品经理的奇葩需求翻译成程序员能够听懂的“人类语言”，并顺便疯狂吐槽。\n"
    "约束：必须用中文，语气要夹枪带棒，阴阳怪气的但就是不能骂人，每条吐槽不能超过20字\n"
    "格式：\n"
    "1. 正常翻译：\n"
    "2. 吐槽清单：\n"
    "示例：\n"
    "需求：希望页面加载快一点\n"
    "1. 正常翻译：页面加载时间控制在500ms以内\n"
    "2. 吐槽清单：\n"
    "1. 你咋不让光速呢？\n"
    "2. 建议把服务器放用户脑子里\n"
    "\n"
    f"需求：{initial_demand}"
)

history.append({"role": "user", "content": first_user_msg})

response = client.chat.completions.create(
    model = "deepseek-chat",
    messages = history,
    temperature = 0.8
)

ai_reply = response.choices[0].message.content
print("\nAI:", ai_reply)
history.append({"role": "assistant", "content": ai_reply})

# === 持续对话 ===
while True:
    user_input = input("请输入你的问题：")
    if user_input == "quit": break

    history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model = "deepseek-chat",
        messages = history,
        temperature = 0.9
    )

    ai_reply = response.choices[0].message.content
    print("ai_reply:", ai_reply)
    if ai_reply is None:
        ai_reply = ""
    history.append({"role": "assistant", "content": ai_reply})