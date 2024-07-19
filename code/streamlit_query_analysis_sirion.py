import os
from utils import *
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

# from dotenv import load_dotenv
# load_dotenv()

# username = os.getenv('USERNAME')
# password = os.getenv('PASSWORD')

# credentials = {"usernames": {username: {"name": username,"password": password}}}

######################################################################################################

######################################################################################################

st.set_page_config(layout="wide",page_title='SQL Evaluation Demo', page_icon="https://www.ibm.com/favicon.ico")

hide_streamlit_style = """
            <style>
            footer {visibility: hidden !important;}
            .image {{
            padding-top: 200px; /* Adjust the padding as needed */
        }}            
        </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)






# Initialize the authenticator
# authenticator = stauth.Authenticate(
#     credentials,
#     "my_app",
#     "auth_cookie",
#     cookie_expiry_days=30
# )
# name, authentication_status, username = authenticator.login()

# if authentication_status:
col1, col2 = st.columns([120,10])  # Adjust column ratios as needed
# authenticator.logout('Logout', 'sidebar', key='unique_key')
st.sidebar.markdown("***")
st.sidebar.markdown("***")
st.sidebar.markdown("***")

st.sidebar.markdown("\n\n\n\nThis dashboard shows a preliminary QUERYCRAFT evaluation results for SQL generation based on the ground truths provided by the `team` and inferencing for different models. No Fine Tuning of models have been performed and default inferece models have been used with prompts and parameters tuning.\n\n\n\n\n\n")

st.sidebar.markdown("***")
st.sidebar.markdown("***")

st.sidebar.markdown("**Powered by**")
st.sidebar.image("https://assets-global.website-files.com/61e7d259b7746e3f63f0b6be/65bb9bc37d5b2f3db376cadd_watsonx.png")    


show_dashboard()


# else:
#     st.write("Please Login into QueryCraft Evaluation Dashboard")
########################################################################################################

    
    
########################################################################################################

    
    
