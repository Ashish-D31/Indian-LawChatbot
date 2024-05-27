from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import PyPDF2
import time
import torch

print('Loading model ...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeddings = HuggingFaceEmbeddings(model_name = 'multi-qa-mpnet-base-dot-v1' , model_kwargs = {'device': device})

print('Processing text ...')
reader = PyPDF2.PdfReader('data/ipc_law.pdf')
text = ''
for page in reader.pages:
    text += page.extract_text()

#split text into chunks
text_splitter = CharacterTextSplitter(
    separator = '\n',
    chunk_size = 512,
    chunk_overlap = 64,
    length_function = len
    )
chunks = text_splitter.split_text(text)

time1 = time.time()
print('Storing embedding ...')
vectordb = FAISS.from_texts(chunks, embeddings)
vectordb.save_local('vectorstore')
print("Time required : ", time.time() - time1)
