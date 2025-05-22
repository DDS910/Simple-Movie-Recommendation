import pandas as pd
import ast

#Data PreProcessing
df = pd.read_csv('tmdb_5000_movies.csv')
df[['genres','title','overview','keywords']].head(5)
df[['genres','title','overview','keywords']].isnull().sum()

df[['genres','title','overview','keywords']].dtypes
#df.dropna(subset=['overview'], inplace=True)
df['overview'] = df['overview'].fillna(' ')


def parse_features(x):
  try:
    return[d['name'] for d in ast.literal_eval(x)]
  except:
    return []

df['genres'] = df['genres'].apply(parse_features)
df['keywords'] = df['keywords'].apply(parse_features)
df['keywords'].head(10)

df['final_text'] = df.apply(lambda row: row['overview'] + " " + " ".join(row['genres']) + " " + " ".join(row['keywords']), axis=1)
df[['final_text', 'title']].head(5)

#Create Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['final_text'])

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(tfidf_matrix)

from sklearn.metrics.pairwise import cosine_similarity
def get_recommendation(title, n_recommendation=5):
  idx = df[df['title'].str.lower() == title.lower()].index[0]

  distances, indices = model_knn.kneighbors(tfidf_matrix[idx], n_neighbors=n_recommendation+1)

  recommend = []

  for i in indices[0][1:]:
    recommend.append(df.iloc[i]['title'])
  return recommend

import joblib

joblib.dump(tfidf, 'vectorizer')
joblib.dump(model_knn, 'model')
joblib.dump(tfidf_matrix, 'matrix')