import json
from typing import Dict, Optional
from models import OllamaModel

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
        
        # 在意图库中查找匹配的意图
        for intent in self.intent_data["intents"]:
            if intent["intent"] == response.strip():
                return intent
        return None
    
    def optimize_prompt(self, user_prompt: str, intent: Dict) -> str:
        # 根据意图优化用户提示词
        if "{content}" in intent["user_prompt"]:
            return intent["user_prompt"].format(content=user_prompt)
        return user_prompt 