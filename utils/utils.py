import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

#st.set_page_config(
#        page_title="Audio features Analysis",
#)


def fn_display_dashboard(page='main'):
    # Streamlit app
    st.title('Song Popularity analysis using song features')
    
    # Display the hyperlink
    st.markdown('[Click here to learn about Spotify Music Feature Data Attributes Details (Data Dictionary)](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)')


    #read data
    
    # Define the mapping
    key_mapping = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
    
    # Sidebar filters
    st.sidebar.title('Filter Criteria')
    #Slider for popularity, energy, dancability selection.
    popularity_range = st.sidebar.slider("Select Popularity range:", 0.0, 100.0, (0.0, 100.0))
    energy_range = st.sidebar.slider("Select Energy Range:", 0.0, 1.0, (0.0, 1.0))
    danceability_range = st.sidebar.slider("Select Danceability Range:", 0.0, 1.0, (0.0, 1.0))
      
    @st.cache_data
    def fetch_and_clean_data():
        df_music = pd.read_csv("dataset-hugging-face.csv")
        df_music = df_music.drop(columns=[df_music.columns[0]])
        # Apply the mapping to create 'key_factor'
        df_music['key'] = df_music['key'].map(key_mapping)
        #df['key'] = pd.Categorical(df['key'], categories=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], ordered=True)
        return df_music

    df = fetch_and_clean_data()
    if page == 'artist':
        df['artists'].fillna('', inplace=True)
        selected_artist = st.text_input('Enter artist name:', '')
    
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
    
    # For Genre display filter by selected Genres
    if page == 'genre':
        # Dropdown for genre selection
        genres = filtered_data['track_genre'].unique()
        #selected_genre = st.selectbox('Select a genre:', genres)
        selected_genres = st.multiselect('Select genres:', genres, default=genres)
        filtered_data = filtered_data[(filtered_data['track_genre'].isin(selected_genres))]
    
    # For Artist display filter by selected Artist
    if page == 'artist':
        if (selected_artist != ''):
            filtered_data = filtered_data[(filtered_data['artists'].str.contains(selected_artist, case=False))]
            #st.write('Filtered Data:', filtered_data)
        
    # Compute descriptive statistics
    stats = filtered_data.describe().drop('count')
    # Display in Streamlit
    st.text("Descriptive Statistics of Song Features (Total Records: " + str("{:,}".format(len(filtered_data))) + ")")
    st.table(stats)

    # Plot
    st.markdown("##### Histogram of Popularity: Shows distribution of popularity of songs and range of popularity and how common a popularity is for the selected set of songs.")
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data['popularity'], bins=10, kde=True)
    plt.title(f'Popularity Distribution of songs')
    plt.xlabel('Popularity')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    # Box plot of popularity vs key
    st.markdown("##### Boxplot of Popularity Vs Key: Shows song's popularity based on the Key used. Box plot shows the most common occurance and outliers.")
    st.markdown("Key: The key the track is in. Integers map to pitches using standard Pitch Class notation.")
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", len(key_mapping))
    sns.boxplot(data=filtered_data, x='key', y='popularity', order=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],palette=palette)
    plt.title('Popularity vs Key')
    plt.xlabel('Key')
    plt.ylabel('Popularity')
    st.pyplot(plt)
    
    # Create a violin plot
    st.markdown("##### Violin Plot of Popularity Vs Key: Shows song's popularity based on the Key used.  This plot shows the most common occurance, outliers and how common is each popularity.")
    st.markdown("Key: The key the track is in. Integers map to pitches using standard Pitch Class notation.")
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
    
    # Box plot of popularity vs time_signature
    st.markdown("##### Box Plot of Popularity Vs time_signature: Shows song's popularity based on the time_signature used.  This plot shows the most common occurance, outliers and how common is each popularity.")
    st.markdown("time_signature: An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4. Range: 3 - 7.")
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", len(key_mapping))
    sns.boxplot(data=filtered_data, x='time_signature', y='popularity',palette=palette)
    plt.title('Popularity vs time_signature')
    plt.xlabel('time_signature')
    plt.ylabel('Popularity')
    st.pyplot(plt)
    
    
    #Plot histogram of all song features
    def create_hostogram_plot(hist_list = [], title ="Histogram of Features (Choose feature you want to see by clicking on the legend)"):
        # Create histograms
        fig = go.Figure()
        for col in histogram_list:
            fig.add_trace(go.Histogram(x=filtered_data[col], name=col, opacity=0.5))
        # Update layout
        fig.update_layout(
            title=title,
            barmode='overlay',
            xaxis_title='Value',
            yaxis_title='Count'
        )
        # Display the plot
        st.plotly_chart(fig)
    #
    #
    st.markdown("##### Histogram of danceability, energy, valence: Shows distribution of danceability, energy, valence of songs and range, how common these features are for the selected set of songs.")
    st.markdown("Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.")
    st.markdown("Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.")
    st.markdown("Valance is a measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry. Range: 0 - 1")
    histogram_list = ["danceability","energy","valence"]
    create_hostogram_plot(histogram_list, "Histogram of Danceability, Energy & Valence")
    
    st.markdown("##### Histogram of speechiness, acousticness, liveness: Shows distribution of speechiness, acousticness, liveness of songs and range, how common these features are for the selected set of songs.")
    st.markdown("Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.")
    st.markdown("acousticness is a confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. Range: 0 - 1.")
    st.markdown("liveness Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.")
    histogram_list = ["speechiness","acousticness","liveness"]
    create_hostogram_plot(histogram_list, "Histogram of Speechiness, Acousticness & Liveness")
    
    st.markdown("##### Histogram of instrumentalness shows distribution of instrumentalness of songs and range, how common these features are for the selected set of songs.")
    st.markdown("instrumentalness Predicts whether a track contains no vocals. 'Ooh' and 'aah' sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly 'vocal'. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.")
    histogram_list = ["instrumentalness"]
    create_hostogram_plot(histogram_list, "Histogram of Instrumentalness")
    
    st.markdown("##### Histogram of loudness shows distribution of loudness of songs and range, how common these features are for the selected set of songs.")
    st.markdown("The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.")
    histogram_list = ["loudness"]
    create_hostogram_plot(histogram_list, "Histogram of Loudness")
    
    st.markdown("##### Histogram of tempo shows distribution of tempo of songs and range, how common these features are for the selected set of songs.")
    st.markdown("The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.")
    histogram_list = ["tempo"]
    create_hostogram_plot(histogram_list,"Histogram of Tempo")
    
    # Count the number of songs per key
    st.markdown("##### Histogram of distribution of of songs by key showing how common songs are for each key.")
    st.markdown("Key: The key the track is in. Integers map to pitches using standard Pitch Class notation.")
    key_count = filtered_data['key'].value_counts().reset_index()
    key_count.columns = ['key', 'count']
    # Plotly horizontal bar chart
    fig = px.bar(key_count, x='count', y='key', orientation='h', title='Number of Songs by Key',color_discrete_sequence=['#ADD8E6'])
    fig.update_layout(xaxis_title='Number of Songs', yaxis_title='Key')
    # Display the plot
    st.plotly_chart(fig)
    
    #Spotify songs and features dataset used in this visualization and popularity analysis.
    st.markdown("##### Spotify songs and features dataset used in this visualization and popularity analysis.")
    st.write('Song features Data:', filtered_data)

    # Drop text columns
    filtered_data = filtered_data.select_dtypes(include=[int, float])


    #Songs and features correlation analysis.
    st.header('Feature Correlation:')
    st.markdown("##### This plot shows the correlation matrix of song features which shows the predective power of each feature.")
    # Calculate the correlation matrix
    correlation_matrix = filtered_data.corr()
    st.table(correlation_matrix)

    st.header('Correlation matrix heat map:')
    # Calculate the correlation matrix
    correlation_matrix = filtered_data.corr().stack().reset_index()
    correlation_matrix.columns = ['Feature1', 'Feature2', 'Correlation']

    # Plotly correlation heatmap
    fig = px.density_heatmap(correlation_matrix, x='Feature1', y='Feature2', z='Correlation', color_continuous_scale='RdBu_r', title='Correlation Matrix',
     width=800,height=800)
    fig.update_layout(xaxis_title='Features', yaxis_title='Features')

    # Display the plot
    st.plotly_chart(fig)
