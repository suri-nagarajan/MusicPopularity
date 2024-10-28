import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API credentials
#SPOTIFY_CLIENT_ID = 'your_spotify_client_id'  # Replace with your Spotify client ID
#SPOTIFY_CLIENT_SECRET = 'your_spotify_client_secret'  # Replace with your Spotify client secret
SPOTIFY_CLIENT_ID = st.text_input('Enter SPOTIFY_CLIENT_ID:', '')
SPOTIFY_CLIENT_SECRET = st.text_input('Enter SPOTIFY_CLIENT_SECRET:', '')

# Setup Spotify authentication
def authenticate_spotify():
    auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

# Function to search for a song and get its audio features
def get_audio_features(sp, song_name):
    # Search for the song by name
    results = sp.search(q=song_name, type='track', limit=1)
    
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        track_id = track['id']
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        
        # Get audio features for the track
        audio_features = sp.audio_features([track_id])[0]
        
        return track_name, artist_name, audio_features
    else:
        return None, None, None

# Streamlit app
st.title("Spotify Song Audio Features")

# Input from user
song_name = st.text_input("Enter a song name:")

if song_name:
    sp = authenticate_spotify()  # Authenticate Spotify API
    
    # Get song details and audio features
    track_name, artist_name, audio_features = get_audio_features (sp, song_name)
    
    if audio_features:
        st.subheader(f"Song: {track_name} by {artist_name}")
        st.write("Artist Name:" + artist_name)
        st.write("Audio Features:")
        st.json(audio_features)
    else:
        st.error("No song found. Please try again with a different song name.")

