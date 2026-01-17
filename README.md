🎬 **Movie Recommendation System using NLP & TF-IDF & Streamlit**

This project implements a content-based movie recommendation system using Natural Language Processing (NLP) techniques. By analyzing movie descriptions with TF-IDF Vectorization and computing cosine similarity via Nearest Neighbors, the system recommends movies that are most similar to a user-selected title.

To enhance usability and accessibility, the recommendation engine is deployed as an interactive web application using Streamlit, featuring user authentication (Login & Sign-Up) and a dedicated recommendation interface.

📌 Project Overview

In this project, we:

• Load and preprocess a movie dataset containing:

  • movie_id, name, description, language, release_year, rating, writer, director, cast, genre

• Perform text preprocessing:

• Text cleaning

• Tokenization

• Stopword removal

• Lemmatization

• Convert movie descriptions into numerical vectors using TF-IDF Vectorizer

• Apply cosine similarity with Nearest Neighbors to identify similar movies

• Build a recommendation pipeline that returns the Top-N most similar movies

• Develop a Streamlit web application with:

• User Login & Sign-Up system

• Secure session-based authentication

• Separate recommendation interface after login

• Evaluate recommendations qualitatively for relevance and interpretability

🌐 **Web Application** (Streamlit)

The project is deployed as a Streamlit web app with the following flow:

🔐 **Authentication Module**

• User Sign-Up with username and password

• User Login using stored credentials

• Persistent user management via file-based database (CSV)

• Automatic redirection to the recommendation page after successful login

🎥 **Recommendation Interface**

• Movie selection via dropdown

• One-click recommendation generation

• Displays Top 5 similar movies

• Expandable movie descriptions for better exploration

• Clean, modern dark-themed UI

📂 **Dataset**

• Source: Kaggle

• Features:

• movie_id, name, description, language, released, rating, writer, director, cast, genre

• Learning Type: Unsupervised learning (similarity-based recommendation)

🛠️ **Technologies Used**

• Python 3.x

• Pandas, NumPy – data processing

• Scikit-learn – TF-IDF Vectorizer, Nearest Neighbors

• NLTK / SpaCy – text preprocessing

• Streamlit – interactive web application

• Matplotlib / Seaborn / WordCloud – data visualization

📊 **Model Selection & Evaluation**

• TF-IDF Vectorizer to represent movie descriptions as weighted feature vectors

• Cosine similarity with Nearest Neighbors for similarity ranking

• Manual evaluation by inspecting recommendations for popular movies

• Emphasis on human interpretability — recommendations should feel relevant and intuitive

📈 **Visualizations**

• Word clouds showing frequent terms per genre

• Distribution of TF-IDF feature weights

• Heatmap of cosine similarity scores

• Recommendation examples:

  • Input movie vs. Top-5 suggested titles

🧭 **Workflow**

Movie Data → Text Preprocessing → TF-IDF Vectorization → Nearest Neighbors Similarity → Recommendation Pipeline → Streamlit Web App → Visualization & Evaluation

💼 **Deliverables**

• Cleaned and preprocessed movie dataset

• Trained TF-IDF + Nearest Neighbors model

• Recommendation function:

get_recommendations("Movie Title")


• Streamlit application with Login & Recommendation pages

• Report showcasing sample recommendations

• Deployment-ready Python scripts / notebooks

🔮 **Future Improvements**

• Hybrid recommendation system (content-based + collaborative filtering)

• User-based personalization using ratings history

• Cloud deployment (AWS / Azure / Streamlit Cloud)

• A/B testing recommendations to measure user engagement

• Advanced NLP models (Word2Vec, FastText, BERT embeddings)
