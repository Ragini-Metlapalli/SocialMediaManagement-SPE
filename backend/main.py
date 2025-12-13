from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import re
from contextlib import asynccontextmanager
from helpers import (
    load_nlp_models,
    extract_caption_features,
    predict_best_time_logic,
    logger  # Import Logger
)

# GLOBAL STATE
# ---------------------------------------------------------
model = None
nlp_pipelines = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load LightGBM/Sklearn Model
    global model
    try:
        model = joblib.load("best_time_model.pkl")
        logger.info("Main Model loaded successfully", extra={"component": "model_loader"})
    except Exception as e:
        logger.error("Failed to load model", extra={"error": str(e)})

    # Load Deep Learning Models
    global nlp_pipelines
    try:
        nlp_pipelines = load_nlp_models()
        logger.info("NLP Models loaded successfully", extra={"component": "nlp_loader"})
    except Exception as e:
        logger.error("Failed to load NLP models", extra={"error": str(e)})

    yield
    # Cleanup if needed
    nlp_pipelines.clear()

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# DATA SCHEMAS
# ---------------------------------------------------------
class PredictionRequest(BaseModel):
    platform: str
    followers: int
    account_age_days: int
    verified: int  # 0 or 1
    media_type: str
    location: str
    caption: str
    cross_platform_spread: int # 0 or 1

class PredictionResponse(BaseModel):
    best_day: int
    best_hour: int
    predicted_engagement: float
    nlp_insights: dict

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.get("/")
def read_root():
    logger.info("Health check endpoint called")
    return {"status": "healthy", "message": "Social Media Engagement API is running with NLP Power"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not model:
        logger.critical("Model not loaded during prediction request")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    logger.info("Received prediction request", extra={
        "platform": request.platform,
        "caption_length": len(request.caption)
    })

    try:
        # 1. Extract NLP Features
        nlp_features = extract_caption_features(request.caption, nlp_pipelines)

        # 2. Find Best Time
        best_day, best_hour, max_eng = predict_best_time_logic(
            model=model,
            req=request,
            nlp_features=nlp_features
        )

        logger.info("Prediction successful", extra={
            "best_day": best_day,
            "best_hour": best_hour,
            "engagement": max_eng
        })

        return {
            "best_day": best_day,
            "best_hour": best_hour,
            "predicted_engagement": max_eng,
            "nlp_insights": nlp_features
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("Prediction failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
