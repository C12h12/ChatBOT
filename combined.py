import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.evaluation import load_evaluator
import nest_asyncio
import random

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

# Evaluation functions
def evaluate_response(response, evaluator_type, model, question):
    criteria_dict = {
        "coherence": "1(poor quality) to 5(high quality)",
            
        "relevance": "1(poor quality) to 5(high quality)",
        "fluency": "1(poor quality) to 5(high quality)"
    }
    evaluator = load_evaluator(evaluator_type, llm=model, criteria=criteria_dict)
    try:
        eval_result = evaluator.evaluate_strings(
            prediction=response,
            input=question
        )
        if eval_result and 'score' in eval_result and eval_result['score'] is not None:
             # Add randomness to individual criteria scores
            for key in ['coherence', 'relevance', 'fluency']:
                if key in eval_result:
                    eval_result[key] = max(0, min(1, eval_result[key] + random.uniform(-0.1, 0.1)))
            # add randomness to total score
            eval_result['score'] = max(0, min(1, eval_result['score'] + random.uniform(-0.1, 0.1)))
        return eval_result
    except Exception as e:
        st.error(f"Evaluation error: {e}")
        return None

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

    # Evaluate responses using "criteria" evaluator
    llama2_quality = evaluate_response(llama2_response, "criteria", mistral_llm, input_text)
    mistral_quality = evaluate_response(mistral_response, "criteria", llama2_llm, input_text)
    st.write(llama2_quality)
    st.write(mistral_quality)

    # Display responses side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🦙 LLaMA 2 Response")
        st.write(llama2_response)
        if llama2_quality and 'score' in llama2_quality and llama2_quality['score'] is not None:
            st.markdown(f"<span style='background-color: green;'>**Quality Score:** {llama2_quality['score'] * 100:.2f}%</span>", unsafe_allow_html=True)
            if 'coherence' in llama2_quality:
                st.write(f"**Coherence Score:** {llama2_quality['coherence'] * 100:.2f}%")
            if 'relevance' in llama2_quality:
                st.write(f"**Relevance Score:** {llama2_quality['relevance'] * 100:.2f}%")
            if 'fluency' in llama2_quality:
                st.write(f"**Fluency Score:** {llama2_quality['fluency'] * 100:.2f}%")
        else:
            st.warning("LLaMA 2 Evaluation score unavailable.")

    with col2:
        st.subheader("🌪️ Mistral Response")
        st.write(mistral_response)
        if mistral_quality and 'score' in mistral_quality and mistral_quality['score'] is not None:
            st.markdown(f"<span style='background-color: green;'>**Quality Score:** {mistral_quality['score'] * 100:.2f}%</span>", unsafe_allow_html=True)
            if 'coherence' in mistral_quality:
                st.write(f"**Coherence Score:** {mistral_quality['coherence'] * 100:.2f}%")
            if 'relevance' in mistral_quality:
                st.write(f"**Relevance Score:** {mistral_quality['relevance'] * 100:.2f}%")
            if 'fluency' in mistral_quality:
                st.write(f"**Fluency Score:** {mistral_quality['fluency'] * 100:.2f}%")
        else:
            st.warning("Mistral Evaluation score unavailable.")