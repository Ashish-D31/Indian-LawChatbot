import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import LlamaCpp
from langchain_core.messages import HumanMessage , AIMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from HtmlTemplates import bot_template , user_template , css
global model ,path
def get_conversation_chain(vectordb):
  llm = LlamaCpp(
    model_path = path,
    temperature = 0.4,
    n_ctx = 2048,
    n_batch = 512,
    n_gpu_layers = -1,
    verbose = True
  )

  memory = ConversationBufferMemory(memory_key='chat_history' , return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
      llm = llm,
      retriever = vectordb.as_retriever(),
      memory = memory,
  )
  return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

st.set_page_config(page_title = "Lawbot", page_icon = "🤖")
st.write(css, unsafe_allow_html=True)

if "conversation" not in st.session_state:
  st.session_state.conversation = None
if "chat_history" not in st.session_state:
  st.session_state.chat_history = None
st.header("Your Personal Law ChatBot ⚖️")

with st.spinner("Loading vector data..."):
    embeddings = HuggingFaceEmbeddings(model_kwargs = {'device': 'cuda'})
    vectordb = FAISS.load_local("vectorstore/", embeddings , allow_dangerous_deserialization=True)

with st.sidebar:
    model = st.selectbox(
        'Choose the LLM version',
        ('Llama-2-7b(Faster)' , 'Llama-2-13b(Can answer more complex queries)'))
if model == 'Llama-2-7b(Faster)':
   path = 'Models/llama-2-7b-chat.Q4_K_M.gguf'
elif model == 'Llama-2-13b(Can answer more complex queries)':
   path = 'Models/llama-2-13b-chat.Q4_K_M.gguf'

user_question = st.text_input("Ask a question :")
st.write("Model is :",model)
if user_question:
  with st.spinner("Generating response ..."):
    handle_userinput(user_question)

st.session_state.conversation = get_conversation_chain(vectordb)