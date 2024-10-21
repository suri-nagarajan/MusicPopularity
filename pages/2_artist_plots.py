import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
 
# Streamlit app
st.title('Popularity Distribution by Artist')

#read data
df = pd.read_csv("dataset-hugging-face.csv")
df['artists'].fillna('', inplace=True)

# Define the mapping
key_mapping = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
#df['key'] = pd.Categorical(df['key'], categories=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], ordered=True)

# Apply the mapping to create 'key_factor'
df['key'] = df['key'].map(key_mapping)


# Multi-select for artist selection
#artists = df['artists'].unique()
#selected_artists = st.multiselect('Select artists:', artists, default=artists)
selected_artist = st.text_input('Enter artist name:', '')
#print(selected_artist)

# Sidebar filters
st.sidebar.title('Filter Criteria')
#Slider for popularity, energy, dancability selection.
popularity_range = st.sidebar.slider("Select Popularity range:", 0.0, 100.0, (0.0, 100.0))
energy_range = st.sidebar.slider("Select Energy Range:", 0.0, 1.0, (0.0, 1.0))
danceability_range = st.sidebar.slider("Select Danceability Range:", 0.0, 1.0, (0.0, 1.0))

# Add a multi-select for keys
# Ensure key_name is ordered correctly
#df['key_name'] = pd.Categorical(df['key'], categories=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], ordered=True)
keys = df['key'].unique()

#Filter for key and mode(major /minor key)
selected_keys = st.sidebar.multiselect('Select keys:', keys, default=keys)
# Multi-select buttons for mode
mode_mapping = {0: "Minor", 1: "Major"}
modes = st.sidebar.multiselect('Select Mode:', options=list(mode_mapping.values()), default=list(mode_mapping.values()))

# Filter data by selected genre, Popularity,Key, mode(major/minor keys) energy, dancability
# Filter the data based on the slider value
filtered_data = df[(df['popularity'] >= popularity_range[0]) 
                    & (df['popularity'] <= popularity_range[1])
                    & (df['key'].isin(selected_keys))
                    & (df['energy'] >= energy_range[0]) 
                    & (df['energy'] <= energy_range[1]) 
                    & (df['danceability'] >= danceability_range[0]) 
                    & (df['danceability'] <= danceability_range[1])
                    & (df['mode'].map(mode_mapping).isin(modes))
                   ]
                   
if (selected_artist != ''):
    filtered_data = filtered_data[(filtered_data['artists'].str.contains(selected_artist, case=False))]
    #st.write('Filtered Data:', filtered_data)
#else:
#    filtered_data = filtered_data

#st.dataframe(df)
#st.line_chart(df)

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['popularity'], bins=10, kde=True)
plt.title(f'Popularity Distribution for Selected Artist')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
st.pyplot(plt)

# Box plot of popularity vs key
plt.figure(figsize=(10, 6))
palette = sns.color_palette("husl", len(key_mapping))
sns.boxplot(data=filtered_data, x='key', y='popularity', order=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],palette=palette)
plt.title('Popularity vs Key')
plt.xlabel('Key')
plt.ylabel('Popularity')
st.pyplot(plt)
    
# Create a violin plot
plt.figure(figsize=(10, 6))
palette = sns.color_palette("husl", len(key_mapping))
sns.violinplot(x='key', y='popularity', data=filtered_data, order=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],palette=palette)
#
# Add labels and title
plt.xlabel('Key')
plt.ylabel('Popularity')
plt.title('Popularity Distribution by Key')
#
# Display the plot
st.pyplot(plt)

#--------------------------------------------------------#
## Plotly distribution plot (violin plot)
#fig = px.violin(filtered_data, x='key', y='popularity', color='key', title='Popularity Distribution by Key', box=True, points='all')
## Display the plot
#st.plotly_chart(fig)
#--------------------------------------------------------#

# Count the number of songs per key
key_count = filtered_data['key'].value_counts().reset_index()
key_count.columns = ['key', 'count']

# Plotly horizontal bar chart
fig = px.bar(key_count, x='count', y='key', orientation='h', title='Number of Songs by Key',color_discrete_sequence=['#ADD8E6'])
fig.update_layout(xaxis_title='Number of Songs', yaxis_title='Key')

# Display the plot
st.plotly_chart(fig)

# Scatter plot of popularity vs tempo
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='tempo', y='popularity')
plt.title('Popularity vs Tempo')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Popularity')
st.pyplot(plt)


st.write('Filtered Data:', filtered_data)






