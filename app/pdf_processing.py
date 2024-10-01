import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import PyPDFLoader
import logging


load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY is not found")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


vector_store = None
ensemble_retriever = None
compression_retriever = None

def load_pdf_file(file_path):
    logger.info("Loading PDF file")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    logger.info("PDF file loaded successfully")
    return documents


def create_vector_store(documents):
    global vector_store, ensemble_retriever, compression_retriever
    logger.info("Creating text splitter")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    logger.info("Splitting documents")
    text_splits = text_splitter.split_documents(documents)
    logger.info("Creating embeddings")
    embeddings = OpenAIEmbeddings()
    logger.info("Creating vector store")
    vectorstore = FAISS.from_documents(text_splits, embeddings)
    logger.info("Saving vector store locally")
    vectorstore.save_local("vector_db")
    logger.info("Vector store created and saved successfully")
    retriever_vectordb = vector_store.as_retriever(search_kwargs={"k": 3})
    keyword_retriever = BM25Retriever.from_documents(text_splits)
    keyword_retriever.k = 3
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb, keyword_retriever], weights=[0.5, 0.5])
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                           base_retriever=ensemble_retriever)
    return vectorstore, ensemble_retriever, compression_retriever


def load_vector_store():
    global vector_store
    if vector_store is None:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local("vector_db", embeddings=embeddings, allow_dangerous_deserialization=True)
    return vector_store


def create_ensemble_retriever(vector_store):
    global ensemble_retriever
    if ensemble_retriever is None:
        retriever_vectordb = vector_store.as_retriever(search_kwargs={"k": 3})
        keyword_retriever = BM25Retriever.from_documents(vector_store.docstore._dict.values())
        keyword_retriever.k = 3
        ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb, keyword_retriever], weights=[0.5, 0.5])
    return ensemble_retriever


def create_compression_retriever(ensemble_retriever):
    global compression_retriever
    if compression_retriever is None:
        compressor = CohereRerank()
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
    return compression_retriever
