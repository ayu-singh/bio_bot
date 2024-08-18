this is an RAG application which lets you talk with 2 chapters of the book "Concepts of biology" which is basically from page no. 89 to 132.
application hosted url is https://biobot-vqpl3pempqrnfdcx8pch87.streamlit.app/

THis application is build using Langchain and Groq models
embedding model = "all-MiniLM-L6-v2"  (hugging face model)
groq model = "mixtral-8x7b-32768"
Vector DB = "FAISS"
splitter = "Simple text splitter" (from langchain)
UI  = "Streamlit"
