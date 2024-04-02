# India-LawChatbot
A chatbot designed to assist users with legal queries. It uses Retrieval-Augmented Generation (RAG) architecture to retrieve context from legal documents and answer queries.

# Downloading LLM
1. Click [Here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf) to download the Llama2 7b model
2. Click [Here](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/blob/main/llama-2-13b-chat.Q4_K_M.gguf) to download the Llama2 13b model
3. Store the files in the Models folder
   
# Running the app
1. Open a terminal where the repository is installed
2. Install all requirements using the command "pip install -r requirements.txt" in terminal
3. Run the ingest.py file to create text embeddings from documents in "Data" folder, the embeddings will be stored in "vectorstore" folder
4. Use the command "streamlit run app.py" in terminal
5. Open "http://localhost:8501/" in browser
6. Select the LLM version and enter a query in the input field
