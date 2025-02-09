import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Load a Question-Answering Model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def healthcare_chatbot(user_input):
    context = """Healthcare includes medical services, medication information, doctor appointments, 
                 symptom analysis, and general well-being guidelines. Always consult a professional 
                 for serious health concerns."""
    
    response = qa_pipeline(question=user_input, context=context)
    return response['answer']

def main():
    st.title("Healthcare Assistant Chatbot")
    
    user_input = st.text_input("How can I assist you today?", "")
    
    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: ", response)
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()
