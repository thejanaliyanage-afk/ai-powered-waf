from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import sys

# 1. Import the tokenizer from our new utils file
from utils import waf_tokenizer

# 2. THE FIX: Trick joblib by ensuring __main__ has access to it exactly as it expects
sys.modules['__main__'].waf_tokenizer = waf_tokenizer

# Initialize the FastAPI app
app = FastAPI(title="WAF AI Inference Engine", version="1.0")

# Load the trained model and vectorizer globally when the app starts
try:
    print("Loading AI Model and Vectorizer...")
    model = joblib.load('ai_engine/waf_logistic_regression.pkl')
    vectorizer = joblib.load('ai_engine/waf_tfidf_vectorizer.pkl')
    print("AI Engine ready.")
except Exception as e:
    print(f"Error loading models: {e}")

# Define the Pydantic model for the incoming Request Payload (API Contract)
class ProxyRequest(BaseModel):
    request_id: str
    timestamp: str
    source_ip: str
    method: str
    path: str
    headers: Dict[str, Any]
    query_parameters: Dict[str, Any]
    body: str

# Define the Pydantic model for the Response Payload (API Contract)
class InferenceResponse(BaseModel):
    request_id: str
    is_malicious: bool
    confidence_score: float
    primary_threat_type: str
    action: str

@app.post("/api/v1/inspect", response_model=InferenceResponse)
async def inspect_traffic(request: ProxyRequest):
    """
    Inspects incoming HTTP traffic for malicious payloads.
    """
    try:
        # Extract the relevant payload data from the request
        payload_parts = []
        if request.query_parameters:
            param_string = "&".join([f"{k}={v}" for k, v in request.query_parameters.items()])
            payload_parts.append(param_string)
        if request.body:
            payload_parts.append(request.body)
            
        combined_payload = " ".join(payload_parts)

        # If there's no payload to inspect, assume safe
        if not combined_payload.strip():
             return InferenceResponse(
                request_id=request.request_id,
                is_malicious=False,
                confidence_score=1.0,
                primary_threat_type="Safe",
                action="ALLOW"
            )

        # Vectorize the payload
        X_infer = vectorizer.transform([combined_payload])

        # Ask the model for a prediction and probability
        prediction = model.predict(X_infer)[0]
        probabilities = model.predict_proba(X_infer)[0]
        
        is_malicious = bool(prediction == 1)
        confidence_score = float(probabilities[prediction])

        # Formulate the response based on the API Contract
        if is_malicious:
            threat_type = "SQLi/XSS Detected" 
            action = "BLOCK"
        else:
            threat_type = "Safe"
            action = "ALLOW"

        return InferenceResponse(
            request_id=request.request_id,
            is_malicious=is_malicious,
            confidence_score=confidence_score,
            primary_threat_type=threat_type,
            action=action
        )

    except Exception as e:
        print(f"Error during inspection: {e}")
        raise HTTPException(status_code=500, detail="Internal AI Engine Error")

@app.get("/api/v1/health")
async def health_check():
    """
    Simple health check endpoint for the proxy to ensure the AI engine is running.
    """
    return {"status": "healthy", "model": "Logistic Regression Baseline"}