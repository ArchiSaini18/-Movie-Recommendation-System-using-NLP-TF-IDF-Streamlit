🎬 **Movie Recommendation System using NLP & TF-IDF & Streamlit**

This project implements a content-based movie recommendation system using Natural Language Processing (NLP) techniques. By analyzing movie descriptions and metadata with TF-IDF Vectorization and applying cosine similarity via Nearest Neighbors, the system recommends movies similar to a user’s selected title.

To enhance usability, the recommendation engine is deployed as an interactive web application using Streamlit, complete with user authentication (Login & Sign-Up) and a dedicated recommendation interface.

📌 Project Overview

In this project, we:

• Load and preprocess a movie dataset containing:
movie_id, name, description, language, release year, rating, writer, director, cast, and genre

• Perform text preprocessing (cleaning, tokenization, stopword removal, lemmatization)

• Convert movie descriptions into numerical representations using TF-IDF Vectorizer

• Apply cosine similarity with Nearest Neighbors to identify movies with similar content

• Build a recommendation pipeline that suggests the Top-N most similar movies

• Develop a Streamlit web application with:

   • User Login & Sign-Up system

   • Secure session-based authentication

   • Separate recommendation page after login

• Evaluate recommendations qualitatively for relevance and interpretability

🌐 **Web Application (Streamlit)**

The project is deployed as a Streamlit web app with the following flow:

🔐 Authentication Module

• User Sign-Up with username and password

• User Login using stored credentials

• Persistent user handling using a database / file storage

• Automatic redirection to the recommendation page after login

🎥 **Recommendation Interface**

• Movie selection via dropdown

• One-click recommendation generation

• Displays Top 5 similar movies

• Expandable movie descriptions for better exploration

• Clean UI with dark theme styling

📂 **Dataset**

• Source: Kaggle

• Features:

• movie_id

• name

• description

• language

• released

• rating

• writer

• director

• cast

• genre

**Target**: Unsupervised learning (similarity-based recommendations)

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

• Manual evaluation by checking recommendations for well-known movies

• Emphasis on human interpretability — recommendations must feel relevant

📈 **Visualizations**

• Word clouds for frequent terms per genre

• Distribution of TF-IDF feature weights

• Heatmap of cosine similarity scores

• Recommendation examples:

• Input movie vs. Top-5 suggested titles

🧭 **Workflow**

Movie Data
→ Text Preprocessing
→ TF-IDF Vectorization
→ Nearest Neighbors Similarity
→ Recommendation Pipeline
→ Streamlit Web App
→ Visualization & Evaluation

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

• Advanced NLP models (Word2Vec, BERT embeddings)
