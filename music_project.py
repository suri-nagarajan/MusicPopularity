import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
#import chartify
import altair as alt
 
# Streamlit app
st.title('Song Popularity analysis using song features')

#read data
df = pd.read_csv("dataset-hugging-face.csv")
df = df.drop(columns=[df.columns[0]])

# Define the mapping
key_mapping = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}

# Apply the mapping to create 'key_factor'
df['key'] = df['key'].map(key_mapping)

# Compute descriptive statistics
stats = df.describe().drop('count')
#stats = stats.drop(columns=[stats.columns[0]])
# Display in Streamlit
st.text("Descriptive Statistics of Song Features (Total Records: " + str("{:,}".format(len(df))) + ")")
st.table(stats)

# Dropdown for genre selection
#-->genres = df['track_genre'].unique()
#selected_genre = st.selectbox('Select a genre:', genres)
#-->selected_genres = st.multiselect('Select genres:', genres, default=genres)

# Multi-select for artist selection
#artists = df['artists'].unique()
#selected_artists = st.multiselect('Select artists:', artists, default=artists)


#st.dataframe(df)
#st.line_chart(df)

# Filter data by selected genre
#filtered_data = df[df['track_genre'] == selected_genre]
#filtered_data = df[df['track_genre'].isin(selected_genres)]
#-->filtered_data = df[(df['track_genre'].isin(selected_genres))] 
                   #& (df['artists'].isin(selected_artists))]

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'], bins=10, kde=True)
plt.title(f'Popularity Distribution for all songs')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
st.pyplot(plt)

# Box plot of popularity vs key
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='key', y='popularity')
plt.title('Popularity vs Key')
plt.xlabel('Key')
plt.ylabel('Popularity')
st.pyplot(plt)

#-------------------------------------------------------------------------------------------#
#ch = chartify.Chart(x_axis_type="categorical")
#ch.plot.boxplot(
#data_frame=df,categorical_columns=["key"],numeric_column="popularity",color_column="key",)
#ch.set_title("Distribution of Popularity by Key")
#ch.axes.set_xaxis_label("Key")
#ch.axes.set_yaxis_label("Popularity")
#ch.show()
#-------------------------------------------------------------------------------------------#

# Plotly distribution plot
fig = px.violin(df, x='key', y='popularity', color='key', title='Popularity Distribution by Key', box=True, points='all')
# Display the plot
st.plotly_chart(fig)

# Count the number of songs per key
key_count = df['key'].value_counts().reset_index()
key_count.columns = ['key', 'count']

# Plotly horizontal bar chart
fig = px.bar(key_count, x='count', y='key', orientation='h', title='Number of Songs by Key',color_discrete_sequence=['#ADD8E6'])
fig.update_layout(xaxis_title='Number of Songs', yaxis_title='Key')

# Display the plot
st.plotly_chart(fig)

# Display the plot

# Plotly box plot
#fig = px.box(df, x='key', y='popularity', title='Popularity vs Key', points='all')

# Display the plot
#st.plotly_chart(fig)

# Scatter plot of popularity vs tempo
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='tempo', y='popularity')
plt.title('Popularity vs Tempo')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Popularity')
st.pyplot(plt)

# Plotly scatter plot
fig = px.scatter(df, x='tempo', y='popularity', title='Popularity vs Tempo', hover_data=['tempo', 'popularity'])

# Display the plot
st.plotly_chart(fig)

# make the chart
c1 = (alt.Chart(df).mark_point().encode(x='tempo',y='popularity',color='key',).interactive())
st.altair_chart(c1, use_container_width=True)


st.write('Song features Data:', df)

# Drop text columns
df = df.select_dtypes(include=[int, float])

st.header('Feature Correlation:')
# Calculate the correlation matrix
correlation_matrix = df.corr()
st.table(correlation_matrix)

st.header('Correlation matrix heat map:')
# Calculate the correlation matrix
correlation_matrix = df.corr().stack().reset_index()
correlation_matrix.columns = ['Feature1', 'Feature2', 'Correlation']

# Plotly correlation heatmap
fig = px.density_heatmap(correlation_matrix, x='Feature1', y='Feature2', z='Correlation', color_continuous_scale='RdBu_r', title='Correlation Matrix',
 width=800,height=800)
fig.update_layout(xaxis_title='Features', yaxis_title='Features')

# Display the plot
st.plotly_chart(fig)




