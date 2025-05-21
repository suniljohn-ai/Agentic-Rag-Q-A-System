# ======= step8: Building streamlit UI =======

import streamlit as st
from src.file import agent

st.title("🧠 Agentic Q&A System")
user_ip = st.text_input("Ask me a Question")
if user_ip:
    response = agent.run(user_ip)
    st.write(response)