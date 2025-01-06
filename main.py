# main.py

import streamlit as st
from utils.auth import authenticate, get_user_fullname, render_sidebar

def main():
    st.set_page_config(page_title="Fraud Detection App", layout="wide")
    
    # Initialize Session State variables
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['username'] = ''
    
    if not st.session_state['authenticated']:
        # Display Login Form
        st.title("📈 Fraud Detection Prediction App")
        st.subheader("🔒 Login to Access the App")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
        
        if submit:
            if authenticate(username, password):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.success("✅ Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid username or password. Please try again.")
    else:
        # User is authenticated; render sidebar
        render_sidebar()
        
        st.write("### Welcome to the Fraud Detection Prediction App!")
        st.write("Use the sidebar to navigate through different sections of the app.")

if __name__ == "__main__":
    main()
