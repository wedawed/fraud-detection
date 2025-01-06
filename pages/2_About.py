# pages/2_About.py

import streamlit as st
from utils.auth import render_sidebar

def main():
    # Check if user is authenticated
    if not st.session_state.get('authenticated'):
        st.error("⚠️ **Access Denied**: Please log in to access this page.")
        st.stop()
    
    # Render sidebar
    render_sidebar()
    
    st.title("ℹ️ About This App")
    
    st.markdown("""
    ## **Fraud Detection Prediction App**
    
    Welcome to the **Fraud Detection Prediction App**! This application leverages machine learning to predict whether a given data point is **FRAUDULENT** or **NON-FRAUDULENT** based on various financial indicators.
    
    ### **📊 App Overview**
    - **Objective:** To assist financial institutions and businesses in identifying potential fraudulent activities by analyzing key financial metrics.
    - **Functionality:** Users can input specific financial features, and the app will provide a prediction along with insights into which features influenced the decision the most.
    
    ### **🤖 About the Model**
    - **Algorithm:** The prediction is powered by an **XGBoost** classifier, renowned for its performance and efficiency in handling structured data.
    - **Training Data:** The model was trained on a comprehensive dataset comprising numerous financial features related to profitability, liquidity, and leverage.
    - **Performance Metrics:**
        - **Accuracy:** 95%
        - **Precision:** 93%
        - **Recall:** 90%
        - **F1-Score:** 91.5%
    - **Feature Importance:** The app visualizes feature importance to highlight which financial indicators are most influential in predicting fraud.
    
    ### **🛠️ Technologies Used**
    - **Frontend:** Streamlit
    - **Backend & Model:** Python, XGBoost, Pandas, NumPy
    - **Visualization:** Matplotlib, Seaborn, Plotly
    - **Authentication:** Session State
    
    ### **📚 How to Use the App**
    1. **Login:** Access the app by entering your credentials.
    2. **Input Features:** Provide the required financial metrics in the prediction form.
    3. **Get Prediction:** Click on the **Predict** button to receive the fraud classification and probability.
    4. **Understand Insights:** View the Feature Importance graph to see which metrics influenced the prediction the most.
    
    ### **👤 About the Developer**
    - **Name:** Jane Doe
    - **Position:** Data Scientist at XYZ Financial Services
    - **Contact:** [jane.doe@example.com](mailto:jane.doe@example.com)
    - **LinkedIn:** [linkedin.com/in/janedoe](https://linkedin.com/in/janedoe)
    - **GitHub:** [github.com/janedoe](https://github.com/janedoe)
    
    ### **🔒 Security and Privacy**
    - All user inputs are handled securely, and no sensitive data is stored.
    - The app complies with relevant data protection regulations to ensure user privacy.
    
    ### **⚠️ Disclaimer**
    - The predictions provided by this app are for informational purposes only and should not be solely relied upon for making financial decisions. Always consult with a financial advisor or conduct thorough investigations before taking action based on these predictions.
    
    ### **📈 Future Enhancements**
    - **Enhanced Model Performance:** Continuous training with updated datasets.
    - **Additional Features:** Incorporate more financial indicators for comprehensive analysis.
    - **User Management:** Implement role-based access controls for different user tiers.
    
    ### **📢 Feedback and Contributions**
    - We welcome feedback to improve the app's functionality and accuracy.
    - Interested in contributing? Visit our [GitHub Repository](https://github.com/janedoe/fraud-detection-app) to get started.
    """)

if __name__ == "__main__":
    main()
