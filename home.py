import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

# Create a page header
st.header("Welcome to our ESRB homepage! 👋")

st.write('This is our ESRB rating exploratory web app. Fell free to have a look around')

    # display a picture
st.image('images/ESRBhome.png')
# Create three columns 
col1, col2, col3 = st.columns([1,1,1])


# inside of the first column
with col1:

    # display the link to that page.
    st.write('<a href="/model_portal"> Check out my Model Portal</a>', unsafe_allow_html=True)
    

#     # display another link to that page
#     st.write('<a href="https://www.behance.net/datatime">View more pretty data visualizations.</a>', unsafe_allow_html=True)


# # inside of column 2
# with col2:

#     # display a link 
#     st.write('<a href="/map"> Check out my Interactive Map</a>', unsafe_allow_html=True)    
    


#     st.write('<a href="https://github.com/zd123"> View more awesome code on my github.</a>', unsafe_allow_html=True)    



# # inside of column 3
# with col3:
#     # st.write('<div style="background:red">asdf </div>', unsafe_allow_html=True)
    
#     # display a link to that page
#     st.write('<a href="/Titanic">Interact with my ML algorithm.</a>', unsafe_allow_html=True)    
    
#     st.markdown('<a href="/Bio">Learn more about me as a human :blush:</a>', unsafe_allow_html=True)


