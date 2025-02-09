from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models import OllamaModel, DeepseekModel
from intent_handler import IntentHandler
from utils import delete_think_tag
app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    model_type: str = "ollama"  # ollama 或 deepseek


@app.post("/chat")
async def chat(request: ChatRequest):
    model = OllamaModel()

    # 意图处理
    intent_handler = IntentHandler()
    intent = intent_handler.recognize_intent(request.message)

    if not intent:
        return {"response": "无法识别意图"}

    # 优化提示词
    final_user_prompt = intent_handler.optimize_prompt(request.message, intent)

    # 生成响应
    response = model.generate(
        system_prompt=intent["final_system_prompt"],
        user_prompt=final_user_prompt
    )

    return {"response": delete_think_tag(response)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
