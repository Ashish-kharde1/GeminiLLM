from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
import sqlite3

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel("gemini-1.5-pro")

def get_gemini_response(question,prompt):
    response=model.generate_content([question,prompt[0]])
    return response.text

def read_sql_query(sql,db):
    connection=sqlite3.connect(db)
    cursor=connection.cursor()
    cursor.execute(sql)
    rows=cursor.fetchall()
    connection.commit()
    connection.close()
    for row in rows:
        print(row)
    return rows


prompt=[
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name STUDENT and has the following columns - NAME, CLASS, 
    SECTION \n\nFor example,\nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM STUDENT ;
    \nExample 2 - Tell me all the students studying in Data Science class?, 
    the SQL command will be something like this SELECT * FROM STUDENT 
    where CLASS="Data Science"; 
    also the sql code should not have ``` in beginning or end and sql word in output.
  

    """

]


st.set_page_config(page_title="I can Retrieve Any SQL query")
st.header("Gemini App To Retrieve SQL Data")

question=st.text_input("Input: ",key="input")

submit=st.button("Ask the question")

if submit:
    response=get_gemini_response(question,prompt)
    print(response)
    sql=read_sql_query(response,"student.db")
    st.subheader("The REsponse is")
    for row in sql:
        print(row)
        st.header(row)
        st.write(row)