import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
 
# Streamlit app
st.title('Popularity Distribution by Artist')

#read data
df = pd.read_csv("dataset-hugging-face.csv")
df['artists'].fillna('', inplace=True)

# Dropdown for genre selection
#genres = df['track_genre'].unique()
#selected_genre = st.selectbox('Select a genre:', genres)
#selected_genres = st.multiselect('Select genres:', genres, default=genres)

# Multi-select for artist selection
#artists = df['artists'].unique()
#selected_artists = st.multiselect('Select artists:', artists, default=artists)
selected_artist = st.text_input('Enter artist name:', '')
#print(selected_artist)
if (selected_artist != ''):
    filtered_data = df[(df['artists'].str.contains(selected_artist, case=False))]
    #st.write('Filtered Data:', filtered_data)
else:
    filtered_data = df

#st.dataframe(df)
#st.line_chart(df)

# Filter data by selected genre
#filtered_data = df[df['track_genre'] == selected_genre]
#filtered_data = df[df['track_genre'].isin(selected_genres)]
#filtered_data = df[(df['artists'].isin(selected_artists))]

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['popularity'], bins=10, kde=True)
plt.title(f'Popularity Distribution for Selected Genres')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
st.pyplot(plt)

# Box plot of popularity vs key
plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_data, x='key', y='popularity')
plt.title('Popularity vs Key')
plt.xlabel('Key')
plt.ylabel('Popularity')
st.pyplot(plt)

# Scatter plot of popularity vs tempo
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='tempo', y='popularity')
plt.title('Popularity vs Tempo')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Popularity')
st.pyplot(plt)


st.write('Filtered Data:', filtered_data)






