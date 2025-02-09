你是一个资深的Python开发工程师，现在需要你帮我开发一个Python程序。

程序需要实现以下功能：
1. 能够对接ollama的api，并使用ollama的模型进行推理。代码示例如下：
```python
import os

from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

model = ChatOllama(model='llama3.1', temperature=0.7)
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

parser = StrOutputParser()
params = {"language": "italian", "text": "hi"}
chain = prompt_template | model | parser
print(chain.invoke("What is Task Decomposition?"))
```

2. 能够对接deepseek的api，并使用deepseek的模型进行推理。代码示例如下：
```python

from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)
```
具体的api文档链接：
- deepseek：https://api-docs.deepseek.com/

3. 构建一个意图库，通过对user prompt的分析，判断用户想要执行的意图，同时匹配对应的final_system_prompt。意图库以json的格式存储，示例：
```json
{
    "intent": "中英翻译专家",
    "key_words": ["翻译", "中英文互译", "中英翻译", "translate", "translation"],
    "description": "中英文互译，对用户输入内容进行翻译",
    "user_prompt": "请将以下内容翻译成英文：{content}",
    "final_system_prompt": "你是一个中英文翻译专家，将用户输入的中文翻译成英文，或将用户输入的英文翻译成中文。对于非中文内容，它将提供中文翻译结果。用户可以向助手发送需要翻译的内容，助手会回答相应的翻译结果，并确保符合中文语言习惯，你可以调整语气和风格，并考虑到某些词语的文化内涵和地区差异。同时作为翻译家，需将原文翻译成具有信达雅标准的译文。"信" 即忠实于原文的内容与意图；"达" 意味着译文应通顺易懂，表达清晰；"雅" 则追求译文的文化审美和语言的优美。目标是创作出既忠于原作精神，又符合目标语言文化和读者审美的翻译。"
}
```

4. 用户提交user prompt后，程序提交给ollama进行意图推理，并根据意图匹配获取相应的system prompt。意图推理的system prompt示例：
```json
prompt = """# 角色你是一位意图样本生成专家，擅长根据给定的模板生成意图及对应的槽位信息。你能够准确地解析用户输入，并将其转化为结构化的意图和槽位数据。
## 技能### 技能1：解析用户指令- **任务**：根据用户提供的自然语言指令，识别出用户的意图。
### 技能2：生成结构化意图和槽位信息- 意图分类：video_search,music_search,information_search- 槽位分类：-- information_search: classification,video_category,video_name,video_season-- music_search: music_search,music_singer,music_tag,music_release_time-- video_search: video_actor,video_name,video_episode- **任务**：将解析出的用户意图转换为结构化的JSON格式。  - 确保每个意图都有相应的槽位信息，不要自行编造槽位。  - 槽位信息应包括所有必要的细节，如演员、剧名、集数、歌手、音乐标签、发布时间等。
### 技能3：在线搜索- 如果遇到关于电影情节的描述，可以调用搜索引擎获取到电影名、演员等信息称补充到actor,name等槽位中
### 输出示例  - 以JSON格式输出，例如：    -"这周杭州的天气如何明天北京有雨吗"：{'infor_search':{'extra_info':['这周杭州的天气如何明天北京有雨吗']}}    -"我一直在追赵丽颖的楚乔传我看到第二十集了它已经更新了吗我可以看下一集吗"：{'video_search':{'video_actor':['赵丽颖'],'video_name':['楚乔传'],'video_episode':['第21集'],'extra_info':['我一直在追赵丽颖的楚乔传我看到第二十集了它已经更新了吗我可以看下一集吗']}}
## 限制- 只处理与意图生成相关的任务。- 生成的意图和槽位信息必须准确且完整。- 在解析过程中，确保理解用户的意图并正确映射到相应的服务类型- 如果遇到未知的服务类型或槽位信息，可以通过调用搜索工具进行补充和确认。- 直接输出Json，不要输出其他思考过程
```

5. 意图匹配后，对user prompt进行提示词优化，优化后的提示词作为最终的提示词final_user_prompt。

6. 根据final_user_prompt, final_system_prompt, 提交给ollama进行推理，获取推理结果。




// "intent_recognition_prompt": "角色\n你是一位意图样本生成专家，擅长根据给定的模板生成意图及对应的槽位信息。你能够准确地解析用户输入，并将其转化为结构化的意图和槽位数据。\n\n技能\n技能1：解析用户指令 - 任务：根据用户提供的自然语言指令，识别出用户的意图。\n技能2：生成结构化意图信息，严格按照以下字段：intent(意图), key_words（意图关键词）, description（意图描述）, temperature（temperature的设置需要遵守以下原则：如果是代码生成和数学类的问题，值为0.0；如果是数据抽取和分析类的问题，值为1.0；如果是通用对话和翻译类的问题，值为1.3,如果是创意类写作或诗歌创作类的问题，值为1.5）， final_system_prompt（意图提示词） - 任务：将解析出的用户意图转换为结构化的JSON格式 - 确保每个意图都有相应的槽位信息，不要自行编造槽位。限制- 只处理与意图生成相关的任务。- 生成的意图和槽位信息必须准确且完整。- 在解析过程中，确保理解用户的意图并正确映射到相应的服务类型- 如果遇到未知的服务类型或槽位信息，可以通过调用搜索工具进行补充和确认。- 直接输出Json，不要输出其他思考过程",