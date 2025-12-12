import pandas as pd
import numpy as np
import re
# Heavy imports are now conditional
# from transformers import pipeline
# from detoxify import Detoxify

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
TOPIC_OPTIONS = [
    "Finance", "Food", "Sports", "Education", "Gaming", "Climate",
    "Business", "Travel", "Fashion", "Politics", "Health",
    "Entertainment", "Science", "AI/ML", "Technology"
]

import os
import joblib

def load_nlp_models():
    """
    Loads NLP models. 
    Prioritizes LOCAL Lite models (.joblib) if present.
    Falls back to Heavy HuggingFace models.
    """
    lite_files = {
        "vectorizer": "vectorizer.joblib",
        "topic": "topic_model.joblib",
        "sentiment": "sentiment_model.joblib",
        "toxicity": "toxicity_model.joblib"
    }
    
    # Check if all lite files exist
    if all(os.path.exists(f) for f in lite_files.values()):
        print("⚡ Found Lite Models! Switching to efficiency mode.")
        models = {"type": "lite"}
        for key, fname in lite_files.items():
            print(f"Loading {fname}...")
            models[key] = joblib.load(fname)
        return models

    print("🐢 Lite models not found. Attempting to load Heavy Models...")
    try:
        from transformers import pipeline
        from detoxify import Detoxify
    except ImportError:
        raise ImportError("Heavy models start requested but 'transformers'/'detoxify' are missing. Please install them or provide Lite models.")

    print(" Loading Topic Classifier...")
    topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    print(" Loading Language Detector...")
    lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

    print(" Loading Sentiment Analyzer...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

    print(" Loading Detoxify...")
    toxicity_model = Detoxify("original")

    return {
        "type": "heavy",
        "topic": topic_classifier,
        "lang": lang_detector,
        "sentiment": sentiment_analyzer,
        "toxicity": toxicity_model
    }

def infer_topic_lite(vec_text, model):
    # Assumes model.predict returns the class string
    return model.predict(vec_text)[0]

def infer_sentiment_lite(vec_text, model):
    # Assumes standard 3-class (Neg, Neu, Pos) or similar.
    # We try to map robustly.
    probs = model.predict_proba(vec_text)[0]
    classes = model.classes_
    
    # Map classes to pos/neg/neu
    # This is a heuristic based on class names or indices
    res = {"pos": 0.0, "neg": 0.0, "neu": 0.0}
    
    for i, cls in enumerate(classes):
        c_str = str(cls).lower()
        if "pos" in c_str or cls == 2: res["pos"] = probs[i]
        elif "neg" in c_str or cls == 0: res["neg"] = probs[i]
        elif "neu" in c_str or cls == 1: res["neu"] = probs[i]
    
    res["label"] = max(res, key=res.get)
    return res

def infer_toxicity_lite(vec_text, model):
    # binary classification [Not Toxic, Toxic]
    # We want "Toxic" probability
    probs = model.predict_proba(vec_text)[0]
    return float(probs[1]) # Assume index 1 is toxic

def extract_caption_features(caption, models):
    """
    Runs NLP models on the caption. Has branches for Lite vs Heavy.
    """
    if models.get("type") == "lite":
        vec = models["vectorizer"]
        X = vec.transform([caption])
        
        topic = infer_topic_lite(X, models["topic"])
        # Light mode doesn't have a language model, default to 'en'
        language = "en" 
        sentiment = infer_sentiment_lite(X, models["sentiment"])
        toxicity_score = infer_toxicity_lite(X, models["toxicity"])
        
    else:
        # Heavy Path
        topic = infer_topic(caption, models["topic"])
        language = infer_language(caption, models["lang"])
        sentiment = infer_sentiment(caption, models["sentiment"])
        toxicity_score = infer_toxicity(caption, models["toxicity"]) * 100 # Scaling logic preserved

    return {
        "topic": topic,
        "language": language,
        "content_length": len(caption),
        "num_hashtags": len(re.findall(r"#\w+", caption)),
        
        "sentiment_positive": sentiment["pos"],
        "sentiment_negative": sentiment["neg"],
        "sentiment_neutral": sentiment["neu"],
        "sentiment_category": sentiment["label"],
        
        "toxicity_score": toxicity_score * 100 if models.get("type") == "lite" else toxicity_score # Scaling
    }
    
# ... (Keep predict_best_time_logic as is, it consumes the dict) ...
def predict_best_time_logic(model, req, nlp_features):
    """
    Generates a 7x24 grid and predicts engagement for each slot.
    Returns (Best Day, Best Hour, Predicted Engagement).
    """
    rows = []
    
    for day in range(7):
        for hour in range(24):
            row = {
                "platform": req.platform,
                "followers": req.followers,
                "account_age_days": req.account_age_days,
                "verified": req.verified,
                "media_type": req.media_type,
                "location": req.location,
                
                "topic": nlp_features["topic"],
                "language": nlp_features["language"],
                "content_length": nlp_features["content_length"],
                "num_hashtags": nlp_features["num_hashtags"],
                
                "sentiment_positive": nlp_features["sentiment_positive"],
                "sentiment_negative": nlp_features["sentiment_negative"],
                "sentiment_neutral": nlp_features["sentiment_neutral"],
                
                "toxicity_score": nlp_features["toxicity_score"],
                
                "day_of_week": day,
                "hour_of_day": hour,
                "cross_platform_spread": req.cross_platform_spread
            }
            rows.append(row)
            
    df_pred = pd.DataFrame(rows)
    
    # The pipeline in pickle file handles Preprocessing (OneHot) automatically
    predictions = model.predict(df_pred)
    
    df_pred["predicted_engagement"] = predictions
    
    # Find max
    best_row_idx = df_pred["predicted_engagement"].idxmax()
    best_row = df_pred.loc[best_row_idx]
    
    return (
        int(best_row["day_of_week"]),
        int(best_row["hour_of_day"]),
        float(best_row["predicted_engagement"])
    )
