from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(input,prompt):
    response=model.generate_content([input,prompt])
    return response.text

st.set_page_config(page_title="Gemini Image Demo")

st.header("Gemini Application")
input=st.text_input("Input Prompt: ",key="input")


submit=st.button("Tell me about the image")

prompt = """
               you are an expert in making blog. You will receive input words as a keyword & you will have to write a blog based on that input keyword
        """


if input:
     with st.spinner("Processing..."):
        response=get_gemini_response(prompt,input)
        st.subheader("The Response is")
        st.write(response)