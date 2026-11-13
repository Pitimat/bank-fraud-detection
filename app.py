
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import joblib

# Modeli yükləyirik
artifact = joblib.load("fraud_detection_model.pkl")

model = artifact["model"]
threshold = artifact["threshold"]
feature_columns = artifact["features"]

app = FastAPI(
    title="Bank Fraud Detection API",
    description="XGBoost-based Fraud Detection System",
    version="1.0"
)


class Transaction(BaseModel):
    features: Dict[str, float]


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "threshold": threshold
    }


@app.post("/predict")
def predict(transaction: Transaction):

    try:
        input_data = transaction.features

        # Çatışmayan feature-ları yoxlayırıq
        missing_features = [
            col for col in feature_columns
            if col not in input_data
        ]

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing_features}"
            )

        # Model üçün DataFrame hazırlayırıq
        df = pd.DataFrame(
            [[input_data[col] for col in feature_columns]],
            columns=feature_columns
        )

        # Fraud probability
        probability = float(
            model.predict_proba(df)[0][1]
        )

        # Classification
        prediction = (
            "Potential Fraud"
            if probability >= threshold
            else "Normal"
        )

        return {
            "fraud_probability": round(probability, 6),
            "prediction": prediction,
            "threshold": threshold
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
