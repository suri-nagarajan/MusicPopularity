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
#SPOTIFY_CLIENT_ID = 'your_spotify_client_id'  # Replace with your Spotify client ID
#SPOTIFY_CLIENT_SECRET = 'your_spotify_client_secret'  # Replace with your Spotify client secret
#SPOTIFY_CLIENT_ID = st.text_input('Enter SPOTIFY_CLIENT_ID:', '')
#SPOTIFY_CLIENT_SECRET = st.text_input('Enter SPOTIFY_CLIENT_SECRET:', '')
SPOTIFY_CLIENT_ID = st.secrets.spotify_credentials.CLIENT_ID
SPOTIFY_CLIENT_SECRET = st.secrets.spotify_credentials.CLIENT_SECRET

#Convert to numeric
def convert_to_numeric(value):
    try:
        return float(value)
    except ValueError:
        return value
        
# Setup Spotify authentication
def authenticate_spotify():
    auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def loadModel(modelName):
    # Load all fragments
    #model = RandomForestClassifier()
    #st.write(modelName)
    with gzip.open(modelName + '.pckl.gz', 'rb') as f:
        model = pickle.load(f)

    
    #for i in range(3):
    #    filename = modelName + '_' + str(i) + '.pckl'
    #    with open(filename, 'rb') as f:
    #        st.write(filename)
    #        if (i == 0):
    #            model_0 = pickle.load(f)
    #            fragment_0 = pickle.dumps(model_0)
    #        if (i == 1):
    #            model_1 = pickle.load(f)
    #            fragment_1 = pickle.dumps(model_1)
    #        if (i == 2):
    #            model_2 = pickle.load(f)
    #            fragment_2 = pickle.dumps(model_2)  
    #        
    ##model = pickle.load(open(modelName + ".pckl", 'rb'))
    ##model.estimators_ = estimators
    #model = pickle.loads(fragment_0 + fragment_1 + fragment_2)
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
        
        # Get the genres of the first artist 
        first_artist_data = sp.artist(first_artist_id) 
        genres = first_artist_data['genres'][0] if first_artist_data['genres'] else ''
        
        return track_name, artist_name, audio_features, genres
    else:
        return None, None, None, None



# Call the function and display the values
#song_features = get_song_features()
#st.write(song_features)


# Streamlit app
st.title("Spotify Song Audio Features")

option = st.radio(
    "How would you like get Song details ?", 
    ('Retrieve from Spotify', 'Experiment by enter it your own values')
    )

if (option == 'Experiment by enter it your own values'):
    # Enter song details
    track_name, artist_name, audio_features, genres  = get_song_features()
    song_name = 'Custom'
    st.write(audio_features)
    
if (option == 'Retrieve from Spotify'):
    # Get song name
    song_name = st.text_input("Enter a song name:")

    if song_name:
        sp = authenticate_spotify()  # Authenticate Spotify API
        
        # Get song details and audio features
        track_name, artist_name, audio_features, genres = get_audio_features (sp, song_name)
        
        if audio_features:
            st.subheader(f"Song: {track_name} by {artist_name}")
            st.write("Artist Name:" + artist_name)
            st.write("Genres :" + genres)
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
        
    #audio_data =  list(audio_data.values())
    # Separate the target and feature variables
    #X = audio_data[all_features]
    #y = audio_data['popularity_level']

    #--->st.write(audio_data)
    
    # Separate the target and feature variables
    audio_data_df = pd.DataFrame([audio_data])
    st.write('Audio features:') #('Audi Data converted to a data frame format')
    st.write(audio_data_df.head())
    X_sample = audio_data_df[all_features]
    
    X_encoded_sample = target_encoder.transform(X_sample)
    X_encoded_sample[numerical_features] = scaler.transform(X_encoded_sample[numerical_features])
    st.write('Audio features after encoding')
    st.write(X_encoded_sample)
    
    rf_model_loaded=loadModel("RandomForestClassifier")
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
    
    # Create the gauge
    level = predicted_popularity
    value = pred_popularity
    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value,
    #title={'text': f'Popularity: {level}'},
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
    
    #st.write(rf_model_loaded.predict(X_encoded_sample))
    
    estimators = rf_model_loaded.estimators_
    tree_predictions = np.array([tree.predict(X_encoded_sample)+1 for tree in estimators])
    tree_predictions_flat = tree_predictions.flatten()
    #st.write(tree_predictions_flat)
    
    # Create the histogram
    #plt.hist(tree_predictions_flat, bins=3, edgecolor='k', alpha=0.7)
    #plt.xticks([1, 2, 3])
    #plt.xlabel('Predicted Values')
    #plt.ylabel('Frequency')
    #plt.title('Histogram of Predictions from All Trees in the RandomForest')
    #st.pyplot(plt)
    
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
        'Importance': feature_importances
    })
    # Plot feature importances
    fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importances')
    st.plotly_chart(fig)
    
    
    
