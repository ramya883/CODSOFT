import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("data_moods.csv")
print("Dataset loaded successfully!")
print(df.head())


features = ["danceability", "energy", "valence", "tempo"]
# Normalize features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
# If mood column missing, create it
if "mood" not in df.columns:
    def assign_mood(valence, energy):
        if valence > 0.6 and energy > 0.6:
            return "Happy"
        elif valence < 0.6 and energy > 0.8:
            return "Energetic"
        elif valence > 0.6 and energy < 0.4:
            return "Calm"
        else:
            return "Sad"
    df["mood"] = df.apply(lambda x: assign_mood(x["valence"], x["energy"]), axis=1)



X = df[features].values
similarity = cosine_similarity(X)
# Function for song-based recommendations
def recommend_by_song(song_name, top_n=5):
    # Case-insensitive search
    matches = df[df['name'].str.lower() == song_name.lower()]
    if matches.empty:
        return pd.DataFrame({"Message": [f" '{song_name}' not found in dataset!"]})
    
    idx = matches.index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return df.iloc[[i[0] for i in sim_scores]][["name", "mood"]]
# Function for mood-based recommendations
def recommend_by_mood(mood, top_n=5):
    if mood not in df['mood'].values:
        return f" Mood '{mood}' not found!"
    mood_songs = df[df['mood'] == mood][["name", "artist", "mood"]]
    if len(mood_songs) < top_n:
        return mood_songs
    else:
        return mood_songs.sample(top_n)



st.title("Mood-Based Music Recommendation System")
st.write("Get song suggestions by **mood** or by **similarity to a song**")

option = st.radio("Choose Recommendation Type:", ["By Mood", "By Song"])

if option == "By Mood":
    mood = st.selectbox("Select a mood:", df['mood'].unique())
    if st.button("Recommend Songs"):
        results = recommend_by_mood(mood)
        st.dataframe(results)

elif option == "By Song":
    song = st.selectbox(" Choose a song:", df['name'].unique())
    if st.button("Find Similar Songs"):
        results = recommend_by_song(song)
        st.dataframe(results)



