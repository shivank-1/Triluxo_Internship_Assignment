#importing libraries
import os
import time
import faiss
import numpy as np
from utils import *
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import streamlit as st
import re
import faiss
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_groq import ChatGroq

#load_dotenv()
extracted_data = load_data()
embeddings = download_embeddings()
llm=ChatGroq(groq_api_key="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",model_name="llama3-70b-8192",temperature=0.6)
dimension = 384  # Dimensionality of Ada 002 embeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
splits = text_splitter.split_documents(extracted_data)
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()
# Initialize session state variables
if 'responses' not in st.session_state:
        st.session_state['responses'] = ["Hi there! Welcome to the Brainlox helpdesk! What can I assist you with today?"]
if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    
# Initialize conversation memory
if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)
    # Define prompt templates
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer as a friendly helpdesk agent and use the provided context to build the answer and If the answer is not contained within the text, say 'I'm not sure about that, but I'm here to help with anything else you need!'. Do not say 'According to the provided context' or anything similar. Just give the answer naturally.""")                                                                        
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
    # Create conversation chain
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

    # Container for chat history
response_container = st.container()
    # Container for text box
text_container = st.container()
with text_container:
        user_query =st.chat_input("Enter your query")
        if user_query:
            with st.spinner("typing..."):
                context = retriever.get_relevant_documents(user_query,k=5)
                response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{user_query}")
            # Append the new query and response to the session state  
            st.session_state.requests.append(user_query)
            st.session_state.responses.append(response)

# Display chat history
with response_container:
      if st.session_state['responses']:
          for i in range(len(st.session_state['responses'])):
              with st.chat_message('Momos',avatar='icon.jpg'):
                  st.write(st.session_state['responses'][i])
              if i < len(st.session_state['requests']):
                  message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

