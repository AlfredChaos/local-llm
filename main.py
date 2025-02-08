from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models import OllamaModel, DeepseekModel
from intent_handler import IntentHandler

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    model_type: str = "ollama"  # ollama 或 deepseek

@app.post("/chat")
async def chat(request: ChatRequest):
    # 选择模型
    if request.model_type == "ollama":
        model = OllamaModel()
    elif request.model_type == "deepseek":
        model = DeepseekModel()
    else:
        raise HTTPException(status_code=400, detail="不支持的模型类型")
    
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
    
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 