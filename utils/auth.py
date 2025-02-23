# utils/auth.py

import bcrypt

# Define users with hashed passwords
# Ensure that the hashed passwords are stored as bytes (prefix with b'')
users = {
    "admin": {
        "name": "Admin User",
        "password": b'$2b$12$kR4hRTuz/PSKdk3EafJMpuCq/LrzzPspZPgXPJKOA786smJkz8nJe'  # hashed 'password123'
    },
    "user1": {
        "name": "User One",
        "password": b'$2b$12$04qZ69PUtXswP7NVHGZPXu76g2ytEAOWUtAL3JRbX3oSw9ANrt9me',  # hashed 'userpassword'
    },
    # Add more users as needed
}

def hash_password(plain_password):
    """
    Hashes a plain text password using bcrypt.
    
    Parameters:
    - plain_password (str): The plain text password to hash.
    
    Returns:
    - bytes: The hashed password.
    """
    return bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt())

def authenticate(username, password):
    """
    Authenticates a user by verifying the provided credentials.
    
    Parameters:
    - username (str): The username entered by the user.
    - password (str): The plain text password entered by the user.
    
    Returns:
    - bool: True if authentication is successful, False otherwise.
    """
    if username in users:
        hashed_pw = users[username]['password']
        return bcrypt.checkpw(password.encode('utf-8'), hashed_pw)
    return False

def get_user_fullname(username):
    """
    Retrieves the full name of the authenticated user.
    
    Parameters:
    - username (str): The username of the authenticated user.
    
    Returns:
    - str: The full name of the user.
    """
    return users.get(username, {}).get('name', 'User')

def logout():
    """
    Logs out the current user by resetting the session state.
    """
    import streamlit as st
    st.session_state['authenticated'] = False
    st.session_state['username'] = ''
    st.experimental_rerun()

def render_sidebar():
    """
    Renders the sidebar with user information and logout button.
    """
    import streamlit as st
    if st.session_state['authenticated']:
        st.sidebar.title("Fraud Detection App")
        st.sidebar.write(f"👤 **{get_user_fullname(st.session_state['username'])}**")
        # Dictionary of pages and their icons
        pages = {
         "Home": "🏠",
         "Prediction": "🔍",
         "About": "ℹ️"
        }
        if st.sidebar.button("🔓 Logout"):
            logout()
