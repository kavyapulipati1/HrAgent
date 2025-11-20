import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="HR Grievance Analyzer",
    page_icon="🤖",
    layout="wide"
)

# --- App Title ---
st.title("🤖 HR Grievance Analyzer")
st.markdown("This tool uses a multi-agent pipeline to analyze an employee grievance, categorize it, find relevant policies, and suggest HR next steps.")

# --- Backend URL ---
# This assumes your FastAPI backend is running on the default port 8000
BACKEND_URL = "http://127.0.0.1:8000/analyze"

# --- User Input ---
st.subheader("📝 Enter Employee Grievance")
grievance_text = st.text_area(
    "Enter the full grievance text below:",
    height=150,
    placeholder="e.g., 'My manager keeps changing my shift timings without any notice and is often rude when I ask about it. I also haven't been paid for my overtime last week.'"
)

if st.button("Analyze Grievance", type="primary"):
    if not grievance_text.strip():
        st.warning("Please enter a grievance to analyze.")
    else:
        with st.spinner("Analyzing... Calling Classifier, Retriever, and Recommender agents..."):
            try:
                # --- API Call ---
                # Call the /analyze endpoint, not the /ask endpoint
                response = requests.post(
                    BACKEND_URL,
                    json={"grievance": grievance_text},
                    timeout=60  # Give it time to run all models
                )

                if response.status_code == 200:
                    # --- Display Results ---
                    data = response.json()

                    st.success("Analysis Complete!")
                    st.divider()

                    # Create two columns for a cleaner layout
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        # --- Recommendation (Main Output) ---
                        st.subheader("💡 Recommended HR Next Steps")
                        st.markdown(data.get("recommendation", "No recommendation provided."))

                    with col2:
                        # --- Categories (from Classifier) ---
                        st.subheader("📊 Detected Categories")
                        categories = data.get("categories", [])
                        if categories:
                            for cat in categories:
                                st.info(f"**{cat}**")
                        else:
                            st.write("No specific categories detected.")

                    st.divider()

                    # --- Policies (from Retriever) ---
                    st.subheader("📚 Retrieved Relevant Policies (Top 3)")
                    policies = data.get("relevant_policies", [])
                    if policies:
                        for i, policy in enumerate(policies, 1):
                            with st.expander(f"**Policy {i}** (Click to see details)"):
                                st.write(policy)
                    else:
                        st.write("No relevant policy documents were found.")

                else:
                    # --- Handle API Errors ---
                    st.error(f"❌ Error from backend: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.RequestException as e:
                # --- Handle Connection Errors ---
                st.error(f"⚠️ Could not connect to backend at {BACKEND_URL}.")
                st.error(f"Please ensure the FastAPI server is running: `uvicorn backend.app:app --reload`")
                print(f"Connection error: {e}")