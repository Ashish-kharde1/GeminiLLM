from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_responce(question):
    model=genai.GenerativeModel("gemini-1.5-flash")
    responce=model.generate_content(question)
    return responce.text

st.set_page_config(page_title="Q&A Demo")
st.header("Gemini LLM application")
input=st.text_input("Input: ", key="input")
submit=st.button("Ask the question")

if submit:
    responce=get_gemini_responce(input)
    st.subheader("The responce is")
    st.write(responce)