import joblib
import pandas as pd
import numpy as np

def inspect(name, path):
    print(f"--- Inspecting {name} ---")
    try:
        obj = joblib.load(path)
        print(f"Type: {type(obj)}")
        if hasattr(obj, "classes_"):
            print(f"Classes: {obj.classes_}")
        if hasattr(obj, "predict_proba"):
            print("Has predict_proba: Yes")
        
        return obj
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

vec = inspect("Vectorizer", "vectorizer.joblib")
tm = inspect("Topic Model", "topic_model.joblib")
sm = inspect("Sentiment Model", "sentiment_model.joblib")
tox = inspect("Toxicity Model", "toxicity_model.joblib")

test_text = ["I love coding with AI!"]
if vec:
    try:
        X = vec.transform(test_text)
        print(f"Vector shape: {X.shape}")
        
        if tm:
            print(f"Topic Pred: {tm.predict(X)}")
        
        if sm:
            print(f"Sentiment Pred: {sm.predict(X)}")
            if hasattr(sm, "predict_proba"):
                print(f"Sentiment Proba: {sm.predict_proba(X)}")
                
        if tox:
            print(f"Toxicity Pred: {tox.predict(X)}")
            if hasattr(tox, "predict_proba"):
                print(f"Toxicity Proba: {tox.predict_proba(X)}")
    except Exception as e:
        print(f"Inference failed: {e}")
