This package contains python code that uses Streamlit to build visualizations of Spotify song audio features and code to execute analytic models that predicts song popularity.
We are providing a song popularity predction engine using Random forest classifier with this project.
You can develop Analytic models of you choice, and export them as a pickle file format and create new model to predict song popularity engine or recommendation engine for this project.

How to run this project:
------------------------
Prerequisite:  
1. Please make sure you installed Python on the machine where you plan to install and run this project.
2. Please make sure all packages in requirements.txt is installed using pip install <package name>


Run this project (on Windows):
------------------------------
1. Copy this package to a local folder on a windows machine.
2. Open windows command prompt.
3. Navigate to the folder where you copied the project.
4. Execute command streamlit run Music_Full_Dataset_Analysis.py
5. This willtake few seconds to bring up a Streamlit server.
6. On you web browser go to url http://localhost:8501

I hosted this App on Streamlit Cloud @ https://musicpopularity-yyejs7ysgnqt8gcgh3f73n.streamlit.app/


Catalog of files and it's functionalities.
------------------------------------------
dataset-hugging-face.csv : CSV file containing 114,000 songs from Spotify.
Music_Full_Dataset_Analysis.py : Python code that displays the main page containing visualizations of entire dataset (dataset-hugging-face.csv).
RandomForestClassifier.pckl.gz : Random Forest classifier analytic model to help predict song popularity. (Note: this is a pickle file which is compressed to help reduce the size so it will fit github file size limit of 100MB).
requirements.txt : This file contains the Python libraries that is needed to run this visualization code. (Note:  You can run pip install requirements.txt to install all dependencies).

In folder utils
utils.py :  This is the common set of streamlit Visualization code that is used by all other visualizations files.  

In folder pages
1_Genre_Analysis.py  : Python code that displays the genere specific filtered visualizations of entire dataset (dataset-hugging-face.csv).
2_Artist_Analysis.py : Python code that displays the Artist specific filtered visualizations of entire dataset (dataset-hugging-face.csv).
3_Popularity_Prediction_and_Experiment_.py : Python code that provides an interface to experiment songs live from Spotify or use our own sonf geatures to predict popularity.
4_Song_Recommendation_Engine.py : Python code for song recommendation engine based in a song we select.

Note on Spotify API interaction credentials: 
--------------------------------------------
Spotify API credentials are needed to interact with Spotify API.
To get SPotify credentilas you must signup for spotify developer account and create an app to generate client ID and client secret which is needed to run this project.
To create Spotify developer account visit : https://developer.spotify.com/
To create Spotify App and generate client id and client secret follow instructions at : https://developer.spotify.com/documentation/web-api

In code 3_Popularity_Prediction_and_Experiment_.py  and 4_Song_Recommendation_Engine.py make following changes.

Option 1: Get Spotify keys from user input (uncomment below lines and comment option 2 to use this option)
SPOTIFY_CLIENT_ID = 'your_spotify_client_id'  # Replace with your Spotify client ID
SPOTIFY_CLIENT_SECRET = 'your_spotify_client_secret'  # Replace with your Spotify client secret
SPOTIFY_CLIENT_ID = st.text_input('Enter SPOTIFY_CLIENT_ID:', '')
SPOTIFY_CLIENT_SECRET = st.text_input('Enter SPOTIFY_CLIENT_SECRET:', '')

 Or

Option 2: Get Spotify keys from Streamlit Secret
To use this option Comment option 1 and in the 'pages' folder create a file 'secrets.toml' with following contents
[spotify_credentials]
CLIENT_ID = "<your_spotify_client_id>"
CLIENT_SECRET = "<Enter SPOTIFY_CLIENT_SECRET>"
POTIFY_CLIENT_ID = st.secrets.spotify_credentials.CLIENT_ID
SPOTIFY_CLIENT_SECRET = st.secrets.spotify_credentials.CLIENT_SECRET



