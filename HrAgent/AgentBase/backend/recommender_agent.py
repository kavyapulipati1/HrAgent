import os
import google.generativeai as genai
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# --- We will use the genai library directly ---
# Configure the library at the module level
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("🤖 Google GenAI library configured successfully!")
else:
    print("⚠️ GEMINI_API_KEY not found. Recommender will run in fallback mode.")

class RecommenderAgent:
    """Generates HR action recommendations using the Google GenAI Python SDK."""

    def __init__(self, api_key: Optional[str] = None):
        
        # ✅ *** FINAL FIX ***
        # We are now using a model name that is confirmed to be in your key's access list.
        self.model_name = "models/gemini-2.5-flash-preview-09-2025"
        
        # Only initialize the model if the key exists
        if GEMINI_API_KEY:
            try:
                self.model = genai.GenerativeModel(self.model_name)
                print(f"✅ Model '{self.model_name}' loaded.")
            except Exception as e:
                print(f"⚠️ Could not load model '{self.model_name}'. Recommender may use fallback. Error: {e}")
                self.model = None
        else:
            self.model = None

    def generate_recommendation(
        self,
        grievance: str,
        categories: List[str],
        policies: List[Dict[str, str]]
    ) -> str:
        """Generate an HR recommendation from grievance, tags, and policy context."""
        
        # If no model or API key, use the local fallback
        if not self.model:
            print("💡 No model loaded. Using offline fallback.")
            return self.fallback(grievance, categories)

        policy_context = "\n\n".join([p["policy_text"] for p in policies]) if policies else "No relevant policies found."

        prompt = f"""
        You are an experienced HR policy assistant.
        An employee has raised the following grievance:

        "{grievance}"

        The detected categories are: {', '.join(categories)}.

        Relevant HR policies:
        {policy_context}

        Based on the above, suggest clear HR next steps or actions.
        Reference policies where possible, and keep your tone formal, brief, and practical.
        """

        try:
            # --- This is the new, cleaner way to call the API ---
            response = self.model.generate_content(prompt)
            
            # Check for safety ratings or other blocks
            if not response.candidates:
                return "⚠️ The model's response was blocked, possibly due to safety settings."

            return response.text

        except Exception as e:
            print(f"⚠️ Gemini SDK error: {e}")
            # If the API call fails, use the local fallback
            return self.fallback(grievance, categories)

    def fallback(self, grievance: str, categories: List[str]) -> str:
        """Offline fallback if Gemini API is unavailable."""
        if not categories:
            return "No categories detected. Please review grievance manually."
        base_actions = []
        for cat in categories:
            if "manager" in cat.lower():
                base_actions.append("Schedule a meeting between the manager and employee to resolve communication issues.")
            elif "payroll" in cat.lower():
                base_actions.append("Verify payroll records and ensure payment discrepancies are addressed.")
            elif "leave" in cat.lower():
                base_actions.append("Check leave records and apply policy-based approval or rejection.")
            elif "harassment" in cat.lower():
                base_actions.append("Launch a confidential inquiry as per the company harassment policy.")
            elif "conduct" in cat.lower():
                base_actions.append("Evaluate the complaint under the code of conduct policy and document findings.")
            else:
                base_actions.append("Forward this grievance to HR for manual escalation.")
        return "Recommended HR actions:\n- " + "\n- ".join(base_actions)