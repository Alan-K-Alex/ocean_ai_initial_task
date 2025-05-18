
from dotenv import load_dotenv
import shutil
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import torch

torch.classes.__path__ = []

import pytesseract  #crucial for performing ocr task (eg:- images maybe present in pdfs)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

load_dotenv(override = True) 

shutil.rmtree("./chroma_db", ignore_errors=True)

# chat_model used to answer users text queries

chat_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    max_tokens=800
)

# the vision_chat_model is used to obtain a text description of an image ,
# which will be stored in vector database

vision_chat_model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    max_retries=2,
    max_tokens=800
)





# the following embedding function is used to convert the data into embeddings 
# before storing into vector database

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}


hf_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)