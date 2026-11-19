
import re
import pickle
import numpy as np

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Project folder path
BASE_DIR = Path(__file__).resolve().parent

# Load trained model
model = load_model(BASE_DIR / "best_model.keras")

# Load tokenizer
with open(BASE_DIR / "tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Create FastAPI application
app = FastAPI(title="SMS Spam Classifier API")

# Input data format
class SMSRequest(BaseModel):
    message: str = Field(..., min_length=1)

# Clean SMS text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Prediction endpoint
@app.post("/predict")
def predict_sms(request: SMSRequest):

    cleaned_message = clean_text(request.message)

    if not cleaned_message:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty"
        )

    sequence = tokenizer.texts_to_sequences([cleaned_message])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=100,
        padding="post"
    )

    probability = float(
        np.asarray(model.predict(padded_sequence, verbose=0)).reshape(-1)[0]
    )

    prediction = "SPAM" if probability >= 0.5 else "HAM"

    return {
        "prediction": prediction,
        "spam_probability": round(probability, 4),
        "confidence": round(
            (probability if prediction == "SPAM" else 1 - probability) * 100,
            2
        )
    }
