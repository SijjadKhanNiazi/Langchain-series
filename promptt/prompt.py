from langchain_core.prompts import PromptTemplate, load_prompt 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
    max_new_tokens=1024, 
    temperature=0.7,
    repetition_penalty=1.1
)
model = ChatHuggingFace (llm=llm)

st.header("Testing Prompts")
st.title("Smart Question Generator")

# User Inputs
user_question = st.text_input("Enter your question here")

tone = st.selectbox(
    "Select Tone",
    ["Formal", "Casual", "Friendly", "Professional", "Humorous"]
)

style = st.selectbox(
    "Select Output Style",
    ["Short Answer", "Detailed Explanation", "Step-by-Step Guide", "Bullet Points"]
)

audience = st.selectbox(
    "Target Audience",
    ["Beginner", "Intermediate", "Expert"]
)

language = st.selectbox(
    "Language",
    ["English", "Urdu"]
)

template = load_prompt("prompt_template.json")
if st.button("Generate Answer"):
    chain = template | model
    result = chain.invoke({
        "user_question": user_question,
        "tone": tone,
        "style": style,
        "audience": audience,
        "language": language
    })
    
    st.write(result.content)

