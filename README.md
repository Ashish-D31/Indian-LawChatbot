# India-LawChatbot
A chatbot designed to assist users with legal queries. It uses Retrieval-Augmented Generation (RAG) architecture to retrieve context from legal documents and answer queries.

# Running the app
1. Open a terminal where the repository is installed
2. Install all requirements using the command "pip install -r requirements.txt" in terminal
3. Run the ingest.py file to create text embeddings from documents in "Data" folder, the embeddings will be stored in "vectorstore" folder
4. Use the command "streamlit run app.py" in terminal
5. Open "http://localhost:8501/" in browser
6. Select the LLM version and enter a query in the input field
