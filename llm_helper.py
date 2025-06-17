from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
load_dotenv()
llm = ChatGroq(groq_api_key=os.getenv("Groq_API_Key"), model_name="llama-3.3-70b-versatile")