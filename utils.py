from langchain_community.document_loaders import WebBaseLoader
import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
from groq import Groq
def load_data():
  loader = WebBaseLoader(web_paths=("https://brainlox.com/courses/category/technical",))
  # Load the content
  docs = loader.load()
  return docs
def download_embeddings():
    embedding_path = "local_embeddings"

    if os.path.exists(embedding_path):
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
    else:
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        with open(embedding_path, 'wb') as f:
            pickle.dump(embedding, f)

    return embedding
