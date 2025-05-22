# Simple-Movie-Recommendation
This project builds a simple movie recommendation system that suggests similar movies based on the descriptive content of each film. The system uses a content-based filtering approach, comparing movie descriptions (overviews), genres, and keywords to measure the similarity between films. The machine learning model used is K-Nearest Neighbors (KNN).

# Dataset
The dataset used in this project is taken from Kaggle. Here is the link: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata.
The dataset used from Kaggle is tmdb_5000_movies.csv
## Important Features for model
1. Title : title of the movies
2. Overview : Description of the movies
3. Genres : list of movies genres
4. Keywords : keyword for the movies

# How the system works
- The data is processed by combining overview, genres, and keywords into a single text.
- TF-IDF is used to convert the text into a numerical representation.
- K-Nearest Neighbors (KNN) is applied to find movies with similar content.
- The results are displayed in a Streamlit application.

# File Explanation
- app.py : Loads the model and provides UI using streamlit for users to choose a movie and see the recommendation
- model : Trained kkn model saved using joblib. Used to find similiar movies
- matrix : Full TF-IDF matrix for all movies, used to similiarity calculation
- vectorizer : Trained TF-IDF vectorizer, used to transform input text into vectors
- Movie_Recommendation.py : Python script that explores the data and build/trains the model 

# How to run
1. Install all the libraries (streamlit, pandas, scikit-learn, joblib)
2. Open your terminal or command prompt and change the directory to where the project is located
3. Start the Streamlit application with: streamlit run app.py 
