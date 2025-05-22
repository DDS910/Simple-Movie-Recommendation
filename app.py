import joblib

model = joblib.load('model')
vectorizer = joblib.load('vectorizer')
matrix = joblib.load('matrix')

import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def load_data():
  df = pd.read_csv('tmdb_5000_movies.csv')
  df['overview'] = df['overview'].fillna(' ')

  def parse_features(x):
    try:
      return[d['name'] for d in ast.literal_eval(x)]
    except:
      return []
    
  df['genres'] = df['genres'].apply(parse_features)
  df['keywords'] = df['keywords'].apply(parse_features)
  df['final_text'] = df.apply(lambda row: row['overview'] + " " + " ".join(row['genres']) + " " + " ".join(row['keywords']), axis=1)

  return df

df = load_data()

def get_recommendation(title, n=5):
  try:
    idx = df[df['title'].str.lower() == title.lower()].index[0]
    distances, indices = model.kneighbors(matrix[idx], n_neighbors=n+1)
    return [df.iloc[i]['title'] for i in indices[0][1:]]
  except:
    return["No movies found"]
  
st.title("Movies Recommendation")

selected = st.selectbox("Choose Movies", sorted(df['title'].unique()))
if st.button("Look recommendation"):
  recs = get_recommendation(selected)
  st.subheader("Movies Recommendation: ")
  for i, r in enumerate(recs, 1):
    st.write(f"{i}. {r}")
  