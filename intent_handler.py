import json
from typing import Dict, Optional
from models import OllamaModel
from utils import delete_think_tag


class IntentHandler:
    def __init__(self):
        with open('intents.json', 'r', encoding='utf-8') as f:
            self.intent_data = json.load(f)
        self.model = OllamaModel()

    def recognize_intent(self, user_prompt: str) -> Optional[Dict]:
        # 使用Ollama进行意图识别
        response = self.model.generate(
            self.intent_data["intent_recognition_prompt"],
            user_prompt
        )
        print("recevice message: ", user_prompt)
        print("intent recognition response: ", response)

        # 清理response中的非json内容
        response = delete_think_tag(response)
        print("response no think: ", response)
        # 查找第一个{和最后一个}的位置
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != 0:
            response = response[start:end]
        new_intent = json.loads(response)

        print("get intent: ", new_intent["intent"])

        # 在意图库中查找匹配的意图
        # 先尝试在现有意图库中查找
        for intent in self.intent_data["intents"]:
            if intent["intent"] == new_intent["intent"]:
                print("意图已匹配")
                return intent
            if new_intent["intent"] in intent["key_words"]:
                print("意图已匹配")
                return intent
            for key_word in new_intent["key_words"]:
                if key_word in intent["key_words"]:
                    print("意图已匹配")
                    return intent

        # 如果找不到,处理response中的json内容
        try:
            print("new intent: ", new_intent)

            # 添加到意图库
            self.intent_data["intents"].append(new_intent)

            # 保存到文件
            with open('intents.json', 'w', encoding='utf-8') as f:
                json.dump(self.intent_data, f, ensure_ascii=False, indent=4)

            return new_intent
        except Exception as e:
            print("error: ", e)
            return None

    def optimize_prompt(self, user_prompt: str, intent: Dict) -> str:
        # 根据意图优化用户提示词
        response = self.model.generate(
            self.intent_data["optimize_prompt"],
            user_prompt
        )
        return delete_think_tag(response)
