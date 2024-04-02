from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
import torch

def make_vectorstores():
    text = ""
    pdf = PdfReader('Data/IPC.pdf','Data/constitution.pdf')
    for page in pdf.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator= '',
        chunk_size = 1024,
        chunk_overlap = 128,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    embeddings = HuggingFaceEmbeddings(model_name = 'multi-qa-mpnet-base-dot-v1' , model_kwargs = {'device': device})

    vectorstore = FAISS.from_texts(chunks , embeddings)
    vectorstore.save_local('vectorstore/')

make_vectorstores()