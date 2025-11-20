HrAgent – Multi-Agent HR Policy Analyzer

HrAgent is an AI-powered system that uses multiple specialized agents to classify, retrieve, and recommend HR policies. It supports ingestion of policy documents, vector-based retrieval, and intelligent response generation using chained agents.

✨ Features

Policy Ingestion – Converts raw HR policy text into embeddings and stores them in ChromaDB.

Classifier Agent – Identifies the type of HR query.

Retriever Agent – Fetches relevant policy chunks using semantic search.

Recommender Agent – Generates final answers or recommendations.

FastAPI Backend – Exposes clean API endpoints.

Embeddings Database (ChromaDB) – Local vector store for fast retrieval.

Frontend-ready structure for UI integration.

📂 Project Structure
HrAgent/
│
├── backend/
│   ├── app.py                # FastAPI main app
│   ├── main.py               # Entry point
│   ├── classifier_agent.py   # Classifies the type of query
│   ├── retriever_agent.py    # Retrieves relevant policies
│   ├── recommender_agent.py  # Generates final response
│   ├── ingest_policies.py    # Loads policies into vector DB
│   ├── policies/             # Raw HR policy files
│   ├── input.json            # Sample input for testing
│   ├── output.json           # Sample output
│
├── chroma_db/                # Vector DB (should be .gitignored)
│
├── frontend/                 # For UI (React or others)
│
├── requirements.txt          # Python dependencies
│
└── .env                      # API keys / configs (should be .gitignored)

🚀 Getting Started
1. Clone Repo
git clone <your-repo-url>
cd HrAgent

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Environment Variables

Create a .env file in the root folder and add your keys:

OPENAI_API_KEY=yourkey


(Any other keys your agents need.)

5. Ingest Policies

Before running the app, load your HR policies into ChromaDB:

python backend/ingest_policies.py

6. Run Backend (FastAPI)
uvicorn backend.app:app --reload


Visit API docs at:

http://localhost:8000/docs

🧪 Testing

You can use input.json as a sample query.
Or send POST requests to:

POST /ask
{
  "question": "What is the leave policy?"
}

📌 TODO / Improvements

Add authentication

Add a proper frontend UI

Improve accuracy with better agent orchestration

Move ChromaDB to cloud

Add test cases

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.
