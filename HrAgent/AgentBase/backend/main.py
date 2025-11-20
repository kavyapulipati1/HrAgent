# backend/main.py
import os
from dotenv import load_dotenv
from backend.classifier_agent import GrievanceClassifier
from backend.retriever_agent import PolicyRetriever
from backend.recommender_agent import RecommenderAgent

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def run_pipeline():
    print("\n🤖 HR Grievance Policy Analyzer — CLI Mode\n")
    
    grievance = input("📝 Enter the employee grievance: ").strip()
    if not grievance:
        print("⚠️ Grievance cannot be empty. Exiting.")
        return

    # Step 1 — Classification
    print("\n🔍 Step 1: Classifying grievance...")
    classifier = GrievanceClassifier()
    categories = [c["label"] for c in classifier.classify(grievance)]
    if categories:
        print(f"✅ Detected Categories: {', '.join(categories)}")
    else:
        print("⚠️ No clear categories detected.")

    # Step 2 — Policy Retrieval
    print("\n📚 Step 2: Retrieving relevant HR policies...")
    retriever = PolicyRetriever()
    policies = retriever.search_policies(grievance, top_k=3)
    if not policies:
        print("⚠️ No relevant policies found.")
    else:
        for i, p in enumerate(policies, 1):
            print(f"\n📄 Policy {i}:")
            print(p['policy_text'][:350] + ("..." if len(p['policy_text']) > 350 else ""))

    # Step 3 — Recommendation Generation
    print("\n💡 Step 3: Generating HR recommendation...")
    recommender = RecommenderAgent(api_key=os.getenv("GEMINI_API_KEY"))
    recommendation = recommender.generate_recommendation(grievance, categories, policies)

    print("\n🎯 FINAL RECOMMENDATION:\n")
    print(recommendation)
    print("\n✅ Process completed successfully.\n")


if __name__ == "__main__":
    run_pipeline()
