import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# Testing and saved changes just in case something happens
# will be removed from git later on!!!

print("hi")

model = pickle.load(open('../Model/ESRB_model.pkl', 'rb'))
print(model)
# table
'''y_pred_df = pd.DataFrame(
    data=[y_pred_proba[0]],
    columns=["E", 'ET', 'M', 'T'])
y_pred_df = y_pred_df.iloc[:, [0, 1, 3, 2]]
st.table(y_pred_df)'''

#pickle save
'''
filename='../Model/ESRB_model.pkl'
file = open(filename,'wb')

pickle.dump(final_model, file)
'''
