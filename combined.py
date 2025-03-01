import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

@st.cache_resource
def load_models():
    """Caches and loads models only once for better performance."""
    return Ollama(model="llama2"), Ollama(model="mistral")

# Load models once
llama2_llm, mistral_llm = load_models()

output_parser = StrOutputParser()
llama2_chain = prompt | llama2_llm | output_parser
mistral_chain = prompt | mistral_llm | output_parser

# Streamlit UI
st.title('⚡ LLAMA2 vs MISTRAL - Ultra-Fast Comparison')
input_text = st.text_input("Enter your question:")

async def get_response(chain, question):
    """Asynchronous function to fetch LLM response."""
    return await asyncio.to_thread(chain.invoke, {"question": question})

if input_text:
    with st.spinner("⏳ Generating responses..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run both models asynchronously
        future_llama2 = loop.create_task(get_response(llama2_chain, input_text))
        future_mistral = loop.create_task(get_response(mistral_chain, input_text))
        
        loop.run_until_complete(asyncio.gather(future_llama2, future_mistral))
        
        llama2_response = future_llama2.result()
        mistral_response = future_mistral.result()

    # Display responses side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🦙 LLaMA 2 Response")
        st.write(llama2_response)
    
    with col2:
        st.subheader("🌪️ Mistral Response")
        st.write(mistral_response)
