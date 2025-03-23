#final code
import asyncio
import random
import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
import nest_asyncio

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

# Function to generate random scores if LLM fails
def get_random_scores():
    coherence = random.randint(1, 5)
    relevance = random.randint(1, 5)
    fluency = random.randint(1, 5)
    overall_score = ((coherence + relevance + fluency) / 15) * 100  # Scale to 100%
    return {
        "coherence": coherence,
        "relevance": relevance,
        "fluency": fluency,
        "score": round(overall_score, 2)  # Round to 2 decimal places
    }

# Function to evaluate response using Mistral
def evaluate_with_mistral(response, question):
    eval_prompt = f"""
    Evaluate the following response based on three criteria: coherence, relevance, and fluency.

    **Question:** {question}

    **Response:** {response}

    **Return ONLY a JSON object. No explanations.**

    ```json
    {{
        "coherence": <numeric_score>,
        "relevance": <numeric_score>,
        "fluency": <numeric_score>,
        "score": <average_score>
    }}
    ```
    """
    try:
        evaluator_llm = Ollama(model="mistral")  # Ensure using the correct model
        eval_result = evaluator_llm.invoke(eval_prompt)

        # Extract only JSON part
        match = re.search(r"\{.*?\}", eval_result, re.DOTALL)
        if match:
            eval_result_json = json.loads(match.group())  # Parse only extracted JSON
        else:
            st.error("Invalid JSON format received. Falling back to random scores.")
            return get_random_scores()

        # Scale the score to 100%
        eval_result_json["score"] = ((eval_result_json["coherence"] +
                                       eval_result_json["relevance"] +
                                       eval_result_json["fluency"]) / 15) * 100
        return eval_result_json
    except Exception as e:
        st.error(f"LLM Evaluation error: {e}")
        return get_random_scores()  # Fallback

# Streamlit UI
st.title('⚡ LLAMA2 vs MISTRAL - Ultra-Fast Comparison')
input_text = st.text_input("Enter your question:")

async def get_response(chain, question):
    """Asynchronous function to fetch LLM response."""
    return await asyncio.to_thread(chain.invoke, {"question": question})

if input_text:
    with st.spinner("⏳ Generating responses..."):
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        future_llama2 = loop.create_task(get_response(llama2_chain, input_text))
        future_mistral = loop.create_task(get_response(mistral_chain, input_text))
        loop.run_until_complete(asyncio.gather(future_llama2, future_mistral))
        llama2_response = future_llama2.result()
        mistral_response = future_mistral.result()

    # Evaluate responses using Mistral (LLM-based) or fallback to random scores
    llama2_quality = evaluate_with_mistral(llama2_response, input_text)
    mistral_quality = evaluate_with_mistral(mistral_response, input_text)

    # Display responses side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🦙 LLaMA 2 Response")
        st.write(llama2_response)
        st.markdown(f"**Coherence:** {llama2_quality['coherence']} / 5")
        st.markdown(f"**Relevance:** {llama2_quality['relevance']} / 5")
        st.markdown(f"**Fluency:** {llama2_quality['fluency']} / 5")
        st.markdown(f"**Overall Score:** {round(llama2_quality['score'], 2)}%")

    with col2:
        st.subheader("🌪️ Mistral Response")
        st.write(mistral_response)
        st.markdown(f"**Coherence:** {mistral_quality['coherence']} / 5")
        st.markdown(f"**Relevance:** {mistral_quality['relevance']} / 5")
        st.markdown(f"**Fluency:** {mistral_quality['fluency']} / 5")
        st.markdown(f"**Overall Score:** {round(mistral_quality['score'], 2)}%")
