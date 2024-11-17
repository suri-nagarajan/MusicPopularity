import streamlit as st
import spotipy
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import matplotlib.pyplot as plt
import json
import pickle
import pprint as pp
from collections import OrderedDict
import gzip
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler, PowerTransformer #, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Spotify API credentials
#Option 1: Get Spotify keys from user input (uncomment below liknes and comment option 2 to use this option)
#SPOTIFY_CLIENT_ID = 'your_spotify_client_id'  # Replace with your Spotify client ID
#SPOTIFY_CLIENT_SECRET = 'your_spotify_client_secret'  # Replace with your Spotify client secret
#SPOTIFY_CLIENT_ID = st.text_input('Enter SPOTIFY_CLIENT_ID:', '')
#SPOTIFY_CLIENT_SECRET = st.text_input('Enter SPOTIFY_CLIENT_SECRET:', '')
#
# Or
#
#Option 2: Get Spotify keys from Streamlit Secret
#To use this option Comment option 1 and in the folder where this file is create a file 'secrets.toml' with following contents
#[spotify_credentials]
#CLIENT_ID = "<your_spotify_client_id>"
#CLIENT_SECRET = "<Enter SPOTIFY_CLIENT_SECRET>"
SPOTIFY_CLIENT_ID = st.secrets.spotify_credentials.CLIENT_ID
SPOTIFY_CLIENT_SECRET = st.secrets.spotify_credentials.CLIENT_SECRET

#Convert to numeric
def convert_to_numeric(value):
    try:
        return float(value)
    except ValueError:
        return value
        
# Setup Spotify authentication so we can call Spotify APIs
def authenticate_spotify():
    auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

#This function loads analytic models
#Note:  We are reading a zip file which the zipped version of pickle file (model.pckl)
#       The reason we are doing this is model files tent to be large in 100's of MB
#       But github has a file size restriction of 100MB max per file.
#
def loadModel(modelName):
    # Load model
    #If Model is larege create a zip file (useful for streamlit cloud) 
    with gzip.open(modelName + '.pckl.gz', 'rb') as f:
        model = pickle.load(f)
    #For local runs we can use normal model files (comment abobe option and uncomment option below).
    #with open(modelName + '.pckl', 'rb') as f:
    #    model = pickle.load(f)
    return model

# Function to get song details
def get_song_features():
    #Track Name
    track_name = 'Custom'
    
    # Input for track genre
    track_genre = st.text_input("Enter the track genre:")

    # Input for artists
    artist_name = st.text_input("Enter the artists:")

    # Slider for time signature
    time_signature = st.slider("Select the time signature:", min_value=3, max_value=7, value=4)

    # Radio button for mode
    mode = st.radio("Select the mode:", options=[0, 1], format_func=lambda x: "Major" if x == 1 else "Minor")

    # Dropdown for key selection
    key_mapping = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
    key = st.selectbox("Select the key:", options=list(key_mapping.keys()), format_func=lambda x: key_mapping[x])

    # Slider for duration in milliseconds (1 minute to 6 minutes) 
    duration_ms = st.slider("Select the duration (ms):", min_value=60000, max_value=360000, step=1000, value=180000)

    # Sliders for additional features
    danceability = st.slider("Danceability:", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    acousticness = st.slider("Acousticness:", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    energy = st.slider("Energy:", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    speechiness = st.slider("Speechiness:", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    instrumentalness = st.slider("Instrumentalness:", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    liveness = st.slider("Liveness:", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    valence = st.slider("Valence:", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    
    # Slider for loudness 
    loudness = st.slider("Loudness (dB):", min_value=-60, max_value=0, step=1, value=-30)
    
    # Radio button for explicit content 
    explicit = st.radio("Is the content explicit?", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False") 
    
    # Slider for tempo in BPM (0 to 250) 
    tempo = st.slider("Tempo (BPM):", min_value=0, max_value=250, step=1, value=120)
    
    # Return the values as a dictionary
    return track_name, artist_name,{
        "time_signature": time_signature,
        "mode": mode,
        "key": key,
        "duration_ms":duration_ms,
        "danceability": danceability,
        "acousticness": acousticness,
        "energy": energy,
        "speechiness": speechiness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "loudness": loudness,
        "explicit":explicit,
        "tempo":tempo
    }, track_genre
    
# Function to search for a song and get its audio features
def get_audio_features(sp, song_name):
    # Search for the song by name
    results = sp.search(q=song_name, type='track', limit=1)
    
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        track_id = track['id']
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        # Get the first artist's ID 
        first_artist_id = track['artists'][0]['id'] 
                
        # Get audio features for the track
        audio_features = sp.audio_features([track_id])[0]
        song_id = track_id
        track = sp.track(song_id)
        popularity = track['popularity']
        #print(f"Popularity: {popularity}")

        
        # Get the genres of the first artist 
        first_artist_data = sp.artist(first_artist_id) 
        genres = first_artist_data['genres'][0] if first_artist_data['genres'] else ''
        
        return track_name, artist_name, audio_features, genres, popularity
    else:
        return None, None, None, None, None



# Streamlit app
st.title("Popularity Prediction and Experiment")

option = st.radio(
    "How would you like get Song details ?", 
    ('Retrieve from Spotify', 'Experiment by entering your own values')
    )

if (option == 'Experiment by entering your own values'):
    # Enter song details
    st.markdown("#### Enter Artist name and Genre and select audio features for your song usng the controls here")
    st.markdown("##### Artist name and Genre are optional, you can leavle them empty if you are exprimenting with features.")
    
    track_name, artist_name, audio_features, genres  = get_song_features()
    song_name = 'Custom'
    st.write(audio_features)
    
if (option == 'Retrieve from Spotify'):
    # Get song name
    st.markdown("#### Enter a song name here (e.g. Hotel California or Tauba Tauba or any song of your choice)")
    st.markdown("#### and hit enter to fetch the song from spotify in realtime and process the song")
    song_name = st.text_input("Enter a song name:")

    if song_name:
        sp = authenticate_spotify()  # Authenticate Spotify API
        
        # Get song details and audio features
        track_name, artist_name, audio_features, genres, spotify_popularity = get_audio_features (sp, song_name)
        
        if audio_features:
            st.subheader(f"Song: {track_name} by {artist_name}")
            st.write("Artist Name:" + artist_name)
            st.write("Genres :" + genres)
            st.write("Spotify Popularity :" + str(spotify_popularity))
            #st.write("Audio Features:")
            #st.json(audio_features)
        else:
            st.error("No song found. Please try again with a different song name.")

if song_name:
    #read data
    df = pd.read_csv("dataset-hugging-face.csv")
    df['artists'].fillna('', inplace=True)

    # Define the mapping
    key_mapping = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
    #df['key'] = pd.Categorical(df['key'], categories=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], ordered=True)
    df['mode_and_key'] = str(df['mode']) + "-" + str(df['key'])
    # Apply the mapping to create 'key_factor'
    df['key'] = df['key'].map(key_mapping)
    
    # Determine popularity level 
    df.loc[((df.popularity >= 0) & (df.popularity <= 35)), "popularity_level" ] = 1
    df.loc[((df.popularity > 35) & (df.popularity <= 70)), "popularity_level" ] = 2
    df.loc[((df.popularity > 70) & (df.popularity <= 100)), "popularity_level" ] = 3
    
    
    # Features for targeted embeddings
    categorical_features = ['track_genre', 'time_signature', 'mode', 'artists', 'mode_and_key']

    numerical_features = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 
                          'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'explicit']
    #                      
    # Combine categorical and numerical features
    all_features = categorical_features + numerical_features
    #
    # Separate the target and feature variables
    X = df[all_features]
    y = df['popularity_level']
    
    # Initialize TargetEncoder for categorical features
    target_encoder = TargetEncoder(cols=categorical_features)
    
    # Fit the encoder and transform the categorical features
    X_encoded = target_encoder.fit_transform(X, y)
    
    # Standardize the numerical features (optional, for scaling the features)
    scaler = StandardScaler()
    X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])



    #with open('./BaggingClassifier.pckl', 'rb') as file: 
    #    model = pickle.load(file)
    
    #st.write(audio_features)
    #st.write(pp.pprint(audio_features))
    audio_data = audio_features
    audio_data['mode_and_key'] = str(audio_data['mode']) + "-" + str(audio_data['key'])
    audio_data['artist'] = artist_name
    audio_data['track_genre'] = genres
    audio_data['explicit'] = 0
    # Drop specified keys 
    keys_to_drop = ['type', 'id', 'uri', 'track_href', 'analysis_url','key'] 
    for key in keys_to_drop: 
        audio_data.pop(key, None)
        
    audio_data = OrderedDict([
    ('track_genre', audio_data['track_genre']),
    ('artists',artist_name),
    ('time_signature', audio_data['time_signature']),
    ('mode', audio_data['mode']),
    ('mode_and_key', audio_data['mode_and_key']),
    ('duration_ms', audio_data['duration_ms']),
    ('danceability', audio_data['danceability']),
    ('energy', audio_data['energy']),
    ('loudness', audio_data['loudness']),
    ('speechiness', audio_data['speechiness']),
    ('acousticness', audio_data['acousticness']),
    ('instrumentalness', audio_data['instrumentalness']),
    ('liveness', audio_data['liveness']),
    ('valence', audio_data['valence']),
    ('tempo', audio_data['tempo']),
    ('explicit', audio_data['explicit'])
    ])
        
    # Separate the target and feature variables
    audio_data_df = pd.DataFrame([audio_data])
    st.write('Audio features:') #('Audi Data converted to a data frame format')
    st.write(audio_data_df.head())
    X_sample = audio_data_df[all_features]
    
    X_encoded_sample = target_encoder.transform(X_sample)
    X_encoded_sample[numerical_features] = scaler.transform(X_encoded_sample[numerical_features])
    st.write('Audio features after encoding')
    st.write(X_encoded_sample)
    
    #This is where we load the Randomclassifier Model by calling loadModel() function.
    rf_model_loaded=loadModel('RandomForestClassifier')
    # Print the model's parameters
    #st.write("Model Parameters:")
    #st.write(rf_model_loaded.get_params())

    # Print the feature importances
    #st.write("\nFeature Importances:")
    #st.write(rf_model_loaded.feature_importances_)
    
    # Drop the 'mode' column 
    X_encoded_sample = X_encoded_sample.drop('mode', axis=1)
    #rf_model_loaded.predict(X_encoded_sample)
    pred_popularity = rf_model_loaded.predict(X_encoded_sample)[0,]
    if pred_popularity == 1.0:
        predicted_popularity = 'Low'
        color = 'red'
    elif pred_popularity == 2.0:
        predicted_popularity = 'Medium'
        color = 'medium'
    else:
        predicted_popularity = 'High'
        color = 'green'
    
    #Predict Probability of each class.
    pred_prob = rf_model_loaded.predict_proba(X_encoded_sample)[0,]
        
    st.write('Song Popularity prediction: ' + predicted_popularity)
    st.write(' ')
    st.write('Probability the song Popularity to be.')
    st.write('Low    :' + str(round(pred_prob[0,] * 100, 2)) + '%')
    st.write('Medium :' + str(round(pred_prob[1,] * 100, 2)) + '%')
    st.write('High   :' + str(round(pred_prob[2,] * 100, 2)) + '%')
    
    st.markdown("##### Song Popularity as predicted by our analytic model and actual Popularity from Spotify user rating.")
    st.markdown("###### Our Model predicts Popularity in range:")
    st.markdown("###### Low (1): 0 to 35, Medium (2): 35 to 70, High(3): 71 to 100")
    
    
    #Create two column plot
    col1, col2 = st.columns(2)
    
    # Create the gauge
    level = predicted_popularity
    value = pred_popularity
    
    with col1:
        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f'Predicted Popularity'},
        gauge={'axis': {'range': [0, 3], 'tickvals': [1, 2, 3], 'ticktext': ["Low", "Medium", "High"]},
               'bar': {'color': "lightblue"},
               'steps': [
                   {'range': [0, 1], 'color': "white"},
                   {'range': [1, 2], 'color': "white"},
                   {'range': [2, 3], 'color': "white"}],
               'threshold': {
                       'line': {'color': "black", 'width': 3},
                       'thickness': 0.50,
                       'value': value}})) 
        
        fig.update_layout(
        title=dict(
            text = f'Popularity: {level}',
            x=0.5,  # X position (0 = left, 1 = right, 0.5 = center)
            y=0.5, # Y position (0 = bottom, 1 = top)
            xanchor='center',  # Horizontal alignment
            yanchor='top',  # Vertical alignment
            font=dict(
                family="Arial", 
                size=16,  # Font size
                color="black"
                )
            )
        )
        # Render the gauge in Streamlit
        st.plotly_chart(fig)
        
    # Create the gauge
    level = predicted_popularity
    value = pred_popularity
    
    # Create the gauge for Spotify popularity
    if (option == 'Retrieve from Spotify'):
        # Determine popularity level 
        if spotify_popularity <= 35:
            spotify_popularity_level = 'Low'
            color = 'red'
        elif spotify_popularity > 35 and spotify_popularity <= 70:
            spotify_popularity_level = 'Medium'
            color = 'yello'
        else:
            spotify_popularity_level = 'High'
            color = 'green'
        
        level = spotify_popularity_level
        value = spotify_popularity
        with col2:
            fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': f'Spotify Popularity'},
            gauge={'axis': {'range': [0, 100], 'tickvals': [35, 70, 100], 'ticktext': ["Low", "Medium", "High"]},
                   'bar': {'color': "lightblue"},
                   'steps': [
                       {'range': [0, 35], 'color': "white"},
                       {'range': [36, 70], 'color': "white"},
                       {'range': [71, 100], 'color': "white"}],
                   'threshold': {
                           'line': {'color': "black", 'width': 3},
                           'thickness': 0.50,
                           'value': value}})) 
            
            fig.update_layout(
            title=dict(
                text = f'Popularity: {level}',
                x=0.5,  # X position (0 = left, 1 = right, 0.5 = center)
                y=0.5, # Y position (0 = bottom, 1 = top)
                xanchor='center',  # Horizontal alignment
                yanchor='top',  # Vertical alignment
                font=dict(
                    family="Arial", 
                    size=16,  # Font size
                    color="black"
                    )
                )
            )
            # Render the gauge in Streamlit
            st.plotly_chart(fig)
    
    #st.write(rf_model_loaded.predict(X_encoded_sample))
    
    estimators = rf_model_loaded.estimators_
    tree_predictions = np.array([tree.predict(X_encoded_sample)+1 for tree in estimators])
    tree_predictions_flat = tree_predictions.flatten()
    #st.write(tree_predictions)
    #st.write(tree_predictions_flat)
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(tree_predictions_flat, bins=3, kde=True)
    plt.xticks([1, 2, 3])
    plt.title(f'Histogram of Predictions from All Trees in the RandomForest')
    plt.xlabel('Predicted Popularity')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    
    # Get Feature Importances
    feature_importances = rf_model_loaded.feature_importances_
    #st.write(feature_importances)
    # Get model parameters
    feature_names = rf_model_loaded.feature_names_in_
    #st.write(feature_names)

    
    # Display Feature Importances
    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances *100
    })
    # Plot feature importances
    fig = px.bar(feature_importance_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h', 
                    title='Feature Importances (in %)')
    st.plotly_chart(fig)
    
    # Create scatter plot using Plotly Express
    # Map numeric predictions to labels
    popularity_map = {1: 'Low', 2: 'Medium', 3: 'High'}
    popularity_colors = {'Low': '#FF6961', 'Medium': '#FFD700', 'High': '#77DD77'}
    # Convert tree predictions to a DataFrame for Plotly
    df = pd.DataFrame({
        'Tree Index': range(1, len(tree_predictions_flat) + 1),
        'Prediction': tree_predictions_flat
    })
    df['Popularity'] = df['Prediction'].map(popularity_map)  # Map numeric predictions to labels

    # Create scatter plot using Plotly Express
    fig = px.scatter(
        df,
        x='Tree Index',
        y='Popularity',
        color='Popularity',
        color_discrete_map=popularity_colors,
        title="Decision Outputs of Each Tree in the Random Forest",
        labels={"Tree Index": "Tree Index", "Popularity": "Predicted Popularity"}
    )

    # Update layout for a cleaner appearance
    fig.update_layout(
        xaxis_title="Tree Index",
        yaxis_title="Predicted Popularity Level",
        yaxis=dict(categoryorder="array", categoryarray=["Low", "Medium", "High"])
    )

    # Render in Streamlit
    st.plotly_chart(fig)

    # Get Out-of-Bag Score
    #ob_score = rf_model_loaded.oob_score_

    # Get Out-of-Bag Decision Function
    #ob_decision_function = rf_model_loaded.oob_decision_function_

    # Display Out-of-Bag Score
    #st.write(f"Out-of-Bag Score: {oob_score}")

    # Display Out-of-Bag Decision Function
    #st.write("Out-of-Bag Decision Function:")
    #st.write(oob_decision_function)