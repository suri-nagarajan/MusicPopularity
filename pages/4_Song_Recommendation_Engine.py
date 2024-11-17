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

def loadModel(modelName,compressed = 'Y'):
    # Load all fragments
    #model = RandomForestClassifier()
    #st.write(modelName)
    #with gzip.open(modelName + '.pckl.gz', 'rb') as f:
    if (compressed == 'Y' or compressed == 'y'):
        with gzip.open(modelName + '.pckl.gz', 'rb') as f:
            model = pickle.load(f)
    else:
        with open(modelName + '.pckl', 'rb') as f:
            model = pickle.load(f)
    return model

#=================================================================#
@st.cache_data
def read_compressed_csv(csv_file):
     csv_data = pd.read_csv(csv_file,compression='gzip')
     return csv_data
     
def build_df_from_spotify(sp_conn, song_name, all_features, numerical_features,target_encoder, scaler, feature_importances_normalized, weighted_kmeans_model):

    from collections import OrderedDict
    

    results = sp_conn.search(q=song_name, type='track', limit=1)
    
    df_song=None
    if results['tracks']['items']:
            track = results['tracks']['items'][0]
            track_id = track['id']
            track_name = track['name']
            artist_name = track['artists'][0]['name']
            # Get the first artist's ID 
            first_artist_id = track['artists'][0]['id'] 
                    
            # Get audio features for the track
            audio_data = sp.audio_features([track_id])[0]
            song_id = track_id
            track = sp.track(song_id)
            popularity = float(track['popularity'])
            

            popularity_label = 1

            if popularity >= 0 and popularity <= 10:
                popularity_label = 1
            elif popularity > 10 and popularity <= 20:
                popularity_label = 2
            elif popularity > 20 and popularity <= 30:
                popularity_label = 3                
            elif popularity > 30 and popularity <= 40:
                popularity_label = 4 
            elif popularity > 40 and popularity <= 50:
                popularity_label = 5 
            elif popularity > 50 and popularity <= 60:
                popularity_label = 6
            elif popularity > 60 and popularity <= 70:
                popularity_label = 7 
            elif popularity > 70 and popularity <= 80:
                popularity_label = 8     
            elif popularity > 80 and popularity <= 90:
                popularity_label = 9 
            else:
                popularity_label = 10
            #st.write("Popularity",popularity,"Label", popularity_label)    
                
            
            # Get the genres of the first artist 
            first_artist_data = sp.artist(first_artist_id) 
            genres = first_artist_data['genres'][0] if first_artist_data['genres'] else ''
            
            #print ( track_name, artist_name, audio_data, genres, popularity)
    
            audio_data['mode_and_key'] = str(audio_data['mode']) + "-" + str(audio_data['key'])
            audio_data['artist'] = artist_name.lower()
            audio_data['track_genre'] = genres.lower()
            audio_data['explicit'] = 0
            audio_data['popularity_label'] = popularity_label
    
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
                ('explicit', audio_data['explicit']),
                ('popularity_label', audio_data['popularity_label'])
                ])
            
            # Separate the target and feature variables
            df_song = pd.DataFrame([audio_data])
            st.markdown("#### Song details from Spotify")
            st.write(df_song)
            X_sample = df_song[all_features ]
            X_encoded_sample = target_encoder.transform(X_sample)
            X_encoded_sample[numerical_features] = scaler.transform(X_encoded_sample[numerical_features])

            X_encoded_sample_weighted = X_encoded_sample.copy()
            X_encoded_sample_weighted.pop('mode')
            
            st.markdown("#### Encoded Song details from Spotify")
            st.write(X_encoded_sample_weighted)
            #X_resampled_weighted.head()
            for k,v in feature_importances_normalized.items():
                #print(k,v)
                X_encoded_sample_weighted[k] = X_encoded_sample_weighted[k]*feature_importances_normalized[k]    
            
            sample_cluster = weighted_kmeans_model.predict(X_encoded_sample_weighted)
            #X_merged = X_encoded_weighted.copy()
            X_encoded_sample_weighted['clusterid'] = pd.Series(sample_cluster)
            X_encoded_sample_weighted['popularity_label'] = popularity_label
            return X_encoded_sample_weighted

def getSimilarSongs2(df, df_joined,source_row, maxcnt):
    from scipy.spatial.distance import cosine
    #80022 is for Kishore Kumar
    #source_row = df.iloc[80022]
    #source_row = df.iloc[iloc]
    source_row = source_row.iloc[0]
    #print('source_row',source_row)
    #cluster = source_row['clusterid'].astype(int)
    cluster = int(source_row['clusterid'])
    #popularity = source_row['popularity_label'].astype(int)
    popularity = int(source_row['popularity_label'])
    st.markdown("#### Song recommendation K-Means cluster search result")
    st.write('cluster=',cluster,'popularity=',popularity)
    
    df2 = df.query("clusterid == @cluster and popularity_label == @popularity").copy()

    cosine_similarities = []
    for i in range(len(df2)):
        #print('xxxx',df2.iloc[i])
        d1= {}
        d1['index']=i
        d1['similarity']=1 - cosine(df2.iloc[i], source_row)
        cosine_similarities.append(d1)

    #sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
    cosine_similarities = sorted(cosine_similarities, key=lambda d: d['similarity'], reverse=True)
    cosine_similarities = cosine_similarities[0:maxcnt]
    r = df_joined.query("clusterid == @cluster and popularity_label == @popularity ")
    locations = [item['index'] for item in cosine_similarities]
    #print('locations',locations, r.shape)
    return r.iloc[locations][['track_genre','artists','album_name','clusterid','popularity_label']]
    #return r.iloc[locations]
    #return cosine_similarities[0:maxcnt]
#==================================================================#

# Streamlit app
st.title("Song Recommendation Engine")

st.markdown("#### This song recommendation engine fetches audio features of the song of your choice from Spotify in realtime and uses K-Means clustering to build recommendations based on the cluster that best matches the song of your choice.")

option = 'Retrieve from Spotify'
#option = st.radio(
#    "How would you like get Song details ?", 
#    ('Retrieve from Spotify', 'Experiment by enter it your own values')
#    )

#if (option == 'Experiment by enter it your own values'):
#    # Enter song details
#    track_name, artist_name, audio_features, genres  = get_song_features()
#    song_name = 'Custom'
#    st.write(audio_features)
    
if (option == 'Retrieve from Spotify'):
    # Get song name
    st.markdown("#### Enter a song name here (e.g. Hotel California or Tauba Tauba or any song of your choice) and hit enter to fetch the song from spotify in realtime and process the song")
    song_name = st.text_input("Enter a song name:")

    if song_name:
        sp = authenticate_spotify()  # Authenticate Spotify API
        
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
        
        #rf_model_loaded=loadModel('RandomForestClassifier','Y')
        rf_model_loaded=loadModel('rfModel-10-level','Y')
        # Get Feature Importances
        #feature_importances = rf_model_loaded.feature_importances_
        feature_names = rf_model_loaded.feature_names_in_
        #st.write(feature_names)
        feature_importances = pd.Series(rf_model_loaded.feature_importances_, index=feature_names).sort_values(ascending=False)
        #st.write(feature_importances)
        feature_importances_normalized = feature_importances/feature_importances['time_signature']
        
        weighted_kmeans_model = loadModel('kmeans-model','N')
    
        # Get song details and audio features
        spotify_track = build_df_from_spotify(sp, song_name, all_features, numerical_features,target_encoder, scaler, feature_importances_normalized, weighted_kmeans_model)
        
        #df = pd.read_csv('X_merged.csv.gzip',compression='gzip')
        df = read_compressed_csv('X_merged.csv.gzip')
        
        #df_joined = pd.read_csv('X_joined.csv.gzip',compression='gzip')
        df_joined = read_compressed_csv('X_joined.csv.gzip')
        
        result = getSimilarSongs2(df, df_joined,spotify_track, maxcnt =10)
        #df.iloc[[102587]]
        
        st.markdown("#### Song recommendations based on your song selection.")
        st.write(result)
        