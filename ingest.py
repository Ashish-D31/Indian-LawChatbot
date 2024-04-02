from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
import torch

def make_vectorstores():
    loader = PyPDFDirectoryLoader('Data/')
    text = ''
    docs = loader.load()
    for page in docs:
        text += page.page_content

    text_splitter = CharacterTextSplitter(
        chunk_size = 1024,
        chunk_overlap = 128,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = HuggingFaceEmbeddings(model_name = 'multi-qa-mpnet-base-dot-v1' , model_kwargs = {'device': device})

    vectorstore = FAISS.from_texts(chunks , embeddings)
    vectorstore.save_local('vectorstore/')

make_vectorstores()