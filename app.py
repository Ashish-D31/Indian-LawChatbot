from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
import streamlit as st
from htmlTemplates import bot_template , user_template , css
import torch

def set_prompt():
    custom_prompt_template = """[INST] <<SYS>>
    You are a trained to guide people about Indian Law. You will answer user's query with your knowledge and the context provided. 
    <</SYS>>
    Use the following pieces of context to answer the users question in short paragraphs.
    Context : {context}
    Question : {question}
    Answer : [/INST]
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 6}),
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_chain

def qa_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = HuggingFaceEmbeddings(model_name = 'multi-qa-mpnet-base-dot-v1' , model_kwargs = {'device': device})

    db = FAISS.load_local("vectorstore", embeddings , allow_dangerous_deserialization=True)

    llm = LlamaCpp(model_path = path , temperature = 0.2 , n_ctx = 2048 , n_batch = 256 , n_gpu_layers = -1 , verbose = False )
    print(path)
    qa_prompt = set_prompt()

    chain = retrieval_qa_chain(llm, qa_prompt, db)
    return chain

def handle_user_input(user_question):
    with st.spinner("Generating response ..."):
            response = chain(user_question)
            response = response['result']
    st.write(bot_template.replace("{{MSG}}",response),unsafe_allow_html=True)
        

st.set_page_config(page_title = "Your personal Law ChatBot", page_icon = ":bot:")
st.write(css , unsafe_allow_html=True)

global chain, path

with st.sidebar:
    model = st.selectbox("Select Model :",("Llama2 7b (Faster)" , "Llama2 13b (Can answer complex queries)"))
    if model == 'Llama2 13b (Can answer complex queries)':
        path = "Models/llama-2-13b-chat.Q4_K_M.gguf"

    elif model == 'Llama2 7b (Faster)':
        path = "Models/llama-2-7b-chat.Q4_K_M.gguf"



if "chat_history" not in st.session_state:
  st.session_state.chat_history = []

st.header("Your personal Law ChatBot :books:")

user_question = st.chat_input("Ask a question :")
with st.spinner("Loading models..."):
    chain = qa_pipeline()

if user_question:
  st.write(user_template.replace("{{MSG}}",user_question),unsafe_allow_html=True)
  handle_user_input(user_question)

for chat in st.session_state.chat_history:
   st.write(user_template.replace("{{MSG}}",chat["User"]),unsafe_allow_html=True)
   st.write(bot_template.replace("{{MSG}}",chat["Bot"]),unsafe_allow_html=True)
