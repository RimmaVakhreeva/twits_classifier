"""
This script sets up a FastAPI application for sentiment analysis of tweets
using machine learning model Mistral.
"""

# Import libraries
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline

import mistral_inference

# Ensure that the necessary environment variables are set
assert "HF_TOKEN" in os.environ
assert "OPENAI_API_KEY" in os.environ

# Dictionary to hold model names and their corresponding inference functions
models = {
    'mistral': (
        pipeline(
            "text-generation",  # Specifying the task type
            model="mistralai/Mistral-7B-Instruct-v0.1",  # Loading the Mistral model
        ), mistral_inference.model_inference),  # Linking to the Mistral inference function
}

app = FastAPI()  # Creating an instance of the FastAPI application


class SentimentRequest(BaseModel):
    """
    Request model for sentiment analysis containing the model name and tweet text.
    """
    model_name: str  # Name of the model to be used for inference
    tweet: str  # The tweet text to analyze


@app.get("/get_sentiment/")
async def get_sentiment(request: SentimentRequest):
    """
    Endpoint to get the sentiment of a tweet using the specified model.

    Args:
        request (SentimentRequest): The request object containing model name and tweet.

    Raises:
        HTTPException: If the model name is invalid or if there is an error during inference.

    Returns:
        JSONResponse: A response containing the tweet and its sentiment.
    """
    # Check if the requested model name is valid
    if request.model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model not found, possible choices are: {list(models.keys())}"  # Provide valid options
        )

    model, inf_func = models[request.model_name]  # Retrieve the model and inference function

    try:
        sentiment = inf_func(model, request.tweet)  # Perform sentiment analysis
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing sentiment while model inference: {str(e)}"  # Handle inference errors
        )

    return JSONResponse(content={
        "tweet": request.tweet,  # Return the original tweet
        "sentiment": sentiment,  # Return the sentiment result
    })


if __name__ == "__main__":
    """
    Entry point for running the FastAPI application.
    The application will be served on host 0.0.0.0 and port 8000.
    """
    import uvicorn  # Importing Uvicorn for serving the FastAPI application

    uvicorn.run(app, host="0.0.0.0", port=8000)  # Running the application
