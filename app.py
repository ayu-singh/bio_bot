from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import faiss
from langchain_community.vectorstores import FAISS
import streamlit as st
import time
import os

groq_api_key = os.environ['GROQ_API_KEY'] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)


document_chain = create_stuff_documents_chain(llm, prompt)


@st.cache_data
def get_embeddings():
    '''
        this function return the embeddings all-MiniLM-L6-v2 model
    '''
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    return embeddings


@st.cache_data
def get_db(_embeddings):
    '''
    this function return the FAISS database containing the embeddings
    '''
    vector_store = FAISS.load_local(
        "faiss_db", _embeddings, allow_dangerous_deserialization=True
    )

    return vector_store


# Streamed response emulator
def response_generator(response):
    '''
    this will give you a streaming response
    '''
    for word in response.split():
        yield word + " "
        time.sleep(0.07)


st.title("I Can answer you Biology Related questions")

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = get_embeddings()

if 'vectors' not in st.session_state:
    st.session_state['vectors'] = get_db(st.session_state['embeddings'])

retriever = st.session_state['vectors'].as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if text := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": text})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(text)
    print(text)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = retrieval_chain.invoke({"input": text})
        print("got response")
        response = st.write_stream(response_generator(response["answer"]))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})