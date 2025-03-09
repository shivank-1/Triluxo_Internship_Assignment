from flask import Flask, request, jsonify, render_template
import os
import time
import faiss
import numpy as np
from utils import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Initialize the components
extracted_data = load_data()
embeddings = download_embeddings()
groq_api_key=os.getenv("Groq_api_key")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0.6)
dimension = 384  # Dimensionality of Ada 002 embeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
splits = text_splitter.split_documents(extracted_data)
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Initialize conversation memory
buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Define prompt templates
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer as a friendly helpdesk agent and use the provided context to build the answer and If the answer is not contained within the text, say 'I'm not sure about that, but I'm here to help with anything else you need!'. Do not say 'According to the provided context' or anything similar. Just give the answer naturally.""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# Create conversation chain
conversation = ConversationChain(memory=buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json['query']
    context = retriever.get_relevant_documents(user_query, k=5)
    response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{user_query}")
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)