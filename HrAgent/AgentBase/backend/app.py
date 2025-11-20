import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.classifier_agent import GrievanceClassifier
from backend.retriever_agent import PolicyRetriever
from backend.recommender_agent import RecommenderAgent # <-- Imports the new recommender
from dotenv import load_dotenv
import requests

# Environment setup
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Initialize FastAPI app
app = FastAPI(title="HR Grievance Policy Agent", version="1.0")

# Initialize agents once at startup
# This will now print "🤖 Google GenAI library configured successfully!"
classifier = GrievanceClassifier()
retriever = PolicyRetriever()
recommender = RecommenderAgent(api_key=os.getenv("GEMINI_API_KEY"))

# Request/Response models
class GrievanceRequest(BaseModel):
    grievance: str

class GrievanceResponse(BaseModel):
    grievance: str
    categories: list[str]
    relevant_policies: list[str]
    recommendation: str

@app.post("/analyze", response_model=GrievanceResponse)
def analyze_grievance(req: GrievanceRequest):
    grievance_text = req.grievance.strip()
    if not grievance_text:
        raise HTTPException(status_code=400, detail="Grievance cannot be empty")
    try:
        # Step 1: Classify grievance
        categories = [c["label"] for c in classifier.classify(grievance_text)]

        # Step 2: Retrieve relevant policies
        policies = retriever.search_policies(grievance_text, top_k=3)

        # Step 3: Generate recommendation (now uses the new SDK-based agent)
        rec = recommender.generate_recommendation(grievance_text, categories, policies)

        return GrievanceResponse(
            grievance=grievance_text,
            categories=categories,
            relevant_policies=[p["policy_text"] for p in policies],
            recommendation=rec,
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "HR Grievance Policy Agent is running 🚀"}

# -----------------------------------------------
# 🕵️ NEW DIAGNOSTIC ENDPOINT
# -----------------------------------------------

@app.get("/list-models")
async def list_models():
    """Lists all available Gemini models for your API key."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        # Filter for only models that support 'generateContent'
        all_models = response.json().get("models", [])
        valid_models = [
            {
                "name": m.get("name"), 
                "supportedGenerationMethods": m.get("supportedGenerationMethods")
            } 
            for m in all_models 
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        
        return {
            "message": "These are the models your API key can use for 'generateContent'.",
            "valid_models": valid_models
        }
        
    except Exception as e:
        print("⚠️ /list-models error:", e)
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")