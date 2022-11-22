# need to remove all the """ strings

# Do not use jupyter Notebook

# takes too long

# load relevant libraries
import numpy as np
import pandas as pd

import streamlit as st

import plotly.express as px

import seaborn as sns

sns.set()




# Import our Random Forest
from sklearn.ensemble import RandomForestClassifier

# Library for visualizing our tree
# If you get an error, run 'conda install python-graphviz' in your terminal (without the quotes).
# import graphviz
# import json
# import datetime

df = pd.read_csv("Video_games_esrb_rating.csv")

df["strong_language"] = df["strong_janguage"]

df.drop("strong_janguage", axis=1, inplace=True)
# dropping the wrongly spelled one

df.drop("console", axis=1, inplace=True)

# dropping console because it is not a useful feature

column_names = list(df.columns)

# no nulls


# we have 33 duplicate rows that we need to remove


df = df.drop_duplicates()

df["num_descriptors"] = 999

# just making a placeholder

list_descriptors = list(df.columns)
list_descriptors.remove("title")

list_descriptors.remove("esrb_rating")
list_descriptors.remove("no_descriptors")
list_descriptors.remove("num_descriptors")

df["num_descriptors"] = df[list_descriptors].sum(axis=1)

df["no_descriptors"] = np.where((df["num_descriptors"] == 0), 1, 0)

encode = {'E': 0,
          'ET': 1,
          'T': 2,
          'M': 3}

df["esrb_encoded"] = df["esrb_rating"].map(encode)

selected_features = list(df.columns)

selected_features.remove("title")
selected_features.remove("esrb_rating")
selected_features.remove("esrb_encoded")

selected_features.remove("no_descriptors")
# removing this as
# num_descriptor is a better version of this
# and makes user input easier


final_model = RandomForestClassifier()

X = df[selected_features]

y = df["esrb_rating"]

final_model.fit(X, y)



# Streamlit App

# Need to use a python file and not an ipynb


st.set_page_config(
    page_title="ESRB",
    page_icon="👋",
)

st.header("Video Game ESRB Prediction Project")

# do not use JupiterNotebook takes forever

# use Thonny and just refresh in the browser


st.write("This is a website made to showcase a model to predict the ESRB ratings of Video Games")

st.subheader('First 5 rows of the data after some cleaning.')

st.dataframe(df[:5])

fig = px.pie(df,
             values=df['esrb_rating'].value_counts(),
             names = df['esrb_rating'].value_counts().index,
             title='Data distribution',
             color=df['esrb_rating'].value_counts().index,
             color_discrete_map={'E': 'lightcyan',
                                 'ET': 'cyan',
                                 'M': 'darkblue',
                                 'T': 'royalblue'}
             )

st.plotly_chart(fig, sharing="streamlit")

st.subheader('Rating Predictor.')

st.write(
    "Below you can enter the descriptors of a potential game and your input will be fed into the model and the prediction will be displayed")

st.write("If you want to see the prediction for a game without any descriptors just hit the predictor button")

# maybe better to use the multiselect as opposed to input,
# but I think this is easier to set up for now


descriptor_list = selected_features.copy()

# needed to copy
# seems otherwise they were pointing to the same thing

descriptor_list.remove("num_descriptors")

user_descriptors = st.multiselect('Descriptors', descriptor_list)



clicked = st.button('Try out the Predictor?')

if (clicked):

    count = len(user_descriptors)

    new_game_values = []

    for descriptor in descriptor_list:

        if (descriptor in user_descriptors):

            new_game_values.append(1)

        else:

            new_game_values.append(0)

    new_game_values.append(count)

    new_game_df = pd.DataFrame([new_game_values], columns=selected_features)

    y_pred = final_model.predict(new_game_df)

    st.write("The model predicted that your game will be")

    st.write('### '+y_pred[0][0])

    y_pred_proba = final_model.predict_proba(new_game_df)

    st.write("The probability for each of the categories are:")
    arr= y_pred_proba[0]



    st.write('- E:  ' + str(round(y_pred_proba[0][0], 2)))
    st.write('- ET: ' + str(round(y_pred_proba[0][1], 2)))
    st.write('- T:  ' + str(round(y_pred_proba[0][3], 2)))
    st.write('- M:  ' + str(round(y_pred_proba[0][2], 2)))

    fig = px.pie(y_pred_proba,
                 names=['E','ET','M','T'],
                 values=y_pred_proba[0],
                 title='ESRB ratings',
                 color=['E','ET','M','T'],
                 color_discrete_map={'E': 'lightcyan',
                                     'ET': 'cyan',
                                     'M': 'darkblue',
                                     'T': 'royalblue'})


    st.plotly_chart(fig, sharing="streamlit")