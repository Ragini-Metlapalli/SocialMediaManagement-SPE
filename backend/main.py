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
import psycopg2
import os

# ---------------------------------------------------------
# DATABASE CONNECTIVITY
# ---------------------------------------------------------
DB_CONNECTION = None

def init_db():
    """
    Connects to Postgres using Standard Environment Variables.
    Secrets are injected by Kubernetes (Fetched from Vault by Jenkins).
    """
    global DB_CONNECTION
    try:
        # Standard Postgres Env Vars (Injected from Vault via K8s Secret)
        USER = os.getenv("DB_USER", "postgres")
        PASSWORD = os.getenv("DB_PASS", "postgres")
        HOST = os.getenv("DB_HOST", "postgres")
        PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME", "prediction_db")

        logger.info(f"Connecting to DB at {HOST}:{PORT} as {USER}...")

        DB_CONNECTION = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            database=DB_NAME
        )
        cursor = DB_CONNECTION.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id SERIAL PRIMARY KEY,
                platform VARCHAR(50),
                caption TEXT,
                predicted_engagement FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        DB_CONNECTION.commit()
        logger.info("✅ Database connected and table verified.")
        cursor.close()
    except Exception as e:
        logger.critical(f"❌ Database connection failed: {str(e)}")
        # We generally don't want to crash the whole app if DB is down, 
        # but for this specific request context, strict failure might be desired.
        # usually just logging error is safer for hybrid setups.


# ---------------------------------------------------------
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

    # Initialize Database
    init_db()

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
        })

        # Log to Database (Async-ish)
        if DB_CONNECTION:
            try:
                cur = DB_CONNECTION.cursor()
                cur.execute(
                    "INSERT INTO prediction_history (platform, caption, predicted_engagement) VALUES (%s, %s, %s)",
                    (request.platform, request.caption, float(max_eng))
                )
                DB_CONNECTION.commit()
                cur.close()
            except Exception as e:
                logger.error("Failed to write to DB", extra={"error": str(e)})

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
