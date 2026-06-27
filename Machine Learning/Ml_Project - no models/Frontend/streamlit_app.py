import streamlit as st
from utils import call_backend_api

st.header("News Category Prediction App 📰")
st.divider()

with st.container():
    st.write("""This Streamlit web application uses machine learning and natural language processing (NLP) to instantly categorize news articles into distinct topics like technology, politics, finance, sports, and entertainment, delivering real-time predictions and visual confidence analytics from raw text inputs.""")

txt = st.text_area("Enter your news article text here:", height=200)

# call backend API to get prediction
if st.button("Predict", type="primary"):
    if txt:
        response = call_backend_api(txt)
        category = response.get("category")
        st.success(f"The Category of the news article is :blue[{category}] :sunglasses: ")
    else:
        st.warning("Please enter some text to predict.")