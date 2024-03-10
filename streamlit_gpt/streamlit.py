from func import create_text
import streamlit as st
from app import model_loaded, tokenizer


st.title('Placebo song generator')
user_input = st.text_input("Add your prompt:")

if st.button('Submit'):
    generated_text = create_text(user_input, model_loaded, tokenizer)
    st.text_area("Result:", value=generated_text, height=250, max_chars=None, key=None)

