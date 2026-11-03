import streamlit as st


def HomePage():
    st.sidebar.page_link("home.py", label="Home Page")

def AboutPage():
    st.sidebar.page_link("about.py", label="About Page")

#display homepage by default somehow 
# Run navigation
st.set_page_config(page_title="KPMG AI Research Agent")

HomePage()
AboutPage()