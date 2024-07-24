import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline

import mistral_inference
import openai_inference

assert "HF_TOKEN" in os.environ
assert "OPENAI_API_KEY" in os.environ

models = {
    'mistral': (
        pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.1",
        ), mistral_inference.model_inference),
    'gpt-4o': (
        "gpt-4o", openai_inference.model_inference
    )
}

app = FastAPI()


class SentimentRequest(BaseModel):
    model_name: str
    tweet: str


@app.get("/get_sentiment/")
async def get_sentiment(request: SentimentRequest):
    if request.model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model not found, possible choices are: {list(models.keys())}"
        )

    model, inf_func = models[request.model_name]

    try:
        sentiment = inf_func(model, request.tweet)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing sentiment while model inference: {str(e)}"
        )

    return JSONResponse(content={
        "tweet": request.tweet,
        "sentiment": sentiment,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
