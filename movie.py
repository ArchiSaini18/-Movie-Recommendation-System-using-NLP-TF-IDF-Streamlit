import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# -------------------------------------------------
# CSS
# -------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #2D3C59; }
html, body, [class*="css"] { color: white !important; }
h1, h2, h3, h4, h5, h6, label, p, span { color: white !important; }

.stTextInput input {
    background-color: white;
    color: black !important;
    border-radius: 8px;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: white !important;
    border-radius: 8px;
}

.stSelectbox div[data-baseweb="select"] span {
    color: #0b1f3a !important;
    font-weight: 600;
}

div[role="listbox"] {
    background-color: #0b1f3a !important;
    border-radius: 8px;
}

div[role="listbox"] span {
    color: white !important;
}

div[role="listbox"] div:hover {
    background-color: #163a66 !important;
}

.stButton > button {
    background-color: #ffcc00;
    color: black !important;
    border-radius: 10px;
    font-weight: bold;
}

.add-movie-section {
    background-color: #1E2A47;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    border: 2px solid #ffcc00;
}

div[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 36px !important;
    font-weight: 800 !important;
}

.predict-section {
    background-color: #1E3A47;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
    border: 2px solid #00ccff;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# User Database (CSV)
# -------------------------------------------------
USER_DB = "users.csv"
MOVIES_DB = "movies_content.csv"

if not os.path.exists(USER_DB):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_DB, index=False)

# Create movies database if it doesn't exist
if not os.path.exists(MOVIES_DB):
    sample_data = {
        'name': ['The Matrix', 'Inception', 'Avatar'],
        'description': [
            'A computer programmer discovers reality is a simulation.',
            'A thief enters people\'s dreams to steal secrets.',
            'A marine fights to save an alien world.'
        ]
    }
    pd.DataFrame(sample_data).to_csv(MOVIES_DB, index=False)

def load_users():
    return pd.read_csv(USER_DB)

def save_user(username, password):
    users = load_users()
    users.loc[len(users)] = [username, password]
    users.to_csv(USER_DB, index=False)

def save_new_movie(name, description):
    """Save a new movie to the database"""
    df = pd.read_csv(MOVIES_DB)
    new_movie = pd.DataFrame({'name': [name.lower()], 'description': [description]})
    df = pd.concat([df, new_movie], ignore_index=True)
    df.to_csv(MOVIES_DB, index=False)
    return True

def predict_movie_recommendations(movie_name, df):
    """Predict recommendations based on movie name only"""
    try:
        # Check if movie exists in database
        if movie_name.lower() in df.name.values:
            # If movie exists, use its description
            movie_description = df.loc[df.name == movie_name.lower(), 'description'].values[0]
        else:
            # If movie doesn't exist, show error
            return None, "Movie not found in database. Please add it first or try a different movie."
        
        # Create TfidfVectorizer with the existing data
        tf = TfidfVectorizer()
        
        # Fit on existing descriptions
        existing_descriptions = df.description.tolist()
        tf.fit(existing_descriptions)
        
        # Transform existing data and movie description
        existing_matrix = tf.transform(existing_descriptions).toarray()
        movie_vector = tf.transform([movie_description]).toarray()
        
        # Train model on existing data
        predict_model = NearestNeighbors(metric="cosine")
        predict_model.fit(existing_matrix)
        
        # Find similar movies
        distances, idx = predict_model.kneighbors(movie_vector, n_neighbors=min(6, len(df)))
        
        # Get recommendations (excluding the movie itself if it's in the list)
        recommendations = []
        for i, distance in zip(idx[0], distances[0]):
            if df.loc[i, 'name'].lower() != movie_name.lower():
                similarity_score = 1 - distance
                recommendations.append({
                    'name': df.loc[i, 'name'].title(),
                    'description': df.loc[i, 'description'],
                    'similarity': similarity_score
                })
        
        return recommendations[:5], None  # Return top 5 recommendations
    
    except Exception as e:
        return None, f"Error generating recommendations: {str(e)}"

# -------------------------------------------------
# Session State
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# -------------------------------------------------
# LOGIN / SIGNUP PAGE
# -------------------------------------------------
def login_page():
    st.title("🎬 Welcome to Movie Recommender")
    users_df = load_users()

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # -------- LOGIN --------
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in users_df.username.values:
                real_pass = users_df.loc[
                    users_df.username == username, "password"
                ].values[0]
                if password == real_pass:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.rerun()
                else:
                    st.error("Wrong password")
            else:
                st.error("User not found")

    # -------- SIGNUP --------
    with tab2:
        new_user = st.text_input("Choose Username")
        new_pass = st.text_input("Choose Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")

        if st.button("Create Account"):
            if new_user in users_df.username.values:
                st.error("Username already exists")
            elif new_pass != confirm:
                st.error("Passwords do not match")
            elif len(new_pass) < 6:
                st.error("Password must be at least 6 characters")
            else:
                save_user(new_user, new_pass)
                st.session_state.logged_in = True
                st.session_state.current_user = new_user
                st.rerun()

# -------------------------------------------------
# MOVIE PAGE
# -------------------------------------------------
def movie_page():
    st.title("🎬 Movie Recommendation System")
    st.write(f"Welcome, **{st.session_state.current_user}**")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()

    @st.cache_data
    def load_data():
        df = pd.read_csv(MOVIES_DB)
        df = df[['name', 'description']]
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['name'] = df['name'].str.lower()
        return df

    @st.cache_resource
    def train_model(df):
        tf = TfidfVectorizer()
        matrix = tf.fit_transform(df.description).toarray()
        model = NearestNeighbors(metric="cosine")
        model.fit(matrix)
        return model, matrix

    # Load data and train model
    df = load_data()
    model, matrix = train_model(df)

    # -------------------------------------------------
    # EXISTING MOVIE SELECTION
    # -------------------------------------------------
    st.header("🎯 Get Recommendations from Existing Movies")
    movie = st.selectbox("Choose a movie from our database:", df.name.str.title())

    if st.button("Get Recommendations", key="existing_movie"):
        mi = df[df.name == movie.lower()].index[0]
        _, idx = model.kneighbors([matrix[mi]], n_neighbors=5)

        st.subheader(f"Movies similar to '{movie}':")
        for i in idx[0][1:]:
            st.write(f"🎥 **{df.loc[i, 'name'].title()}**")
            with st.expander("View Description"):
                st.write(df.loc[i, 'description'])

    # -------------------------------------------------
    # SEPARATOR
    # -------------------------------------------------
    st.markdown("---")

    # -------------------------------------------------
    # ADD NEW MOVIE SECTION
    # -------------------------------------------------
    st.header("➕ Add New Movie")
    
    with st.container():
        st.markdown('<div class="add-movie-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Add Your Movie")
            new_movie_name = st.text_input("Movie Name:", placeholder="e.g., The Dark Knight")
            new_movie_description = st.text_area(
                "Movie Description:", 
                placeholder="Provide a detailed description of the movie's plot, genre, and themes...",
                height=150
            )
        
        # Add movie button (full width)
        st.markdown("---")
        if st.button("Add Movie to Database", key="add_movie"):
            if new_movie_name and new_movie_description:
                if len(new_movie_description) < 50:
                    st.warning("Please provide a more detailed description (at least 50 characters)")
                else:
                    try:
                        # Check if movie already exists
                        if new_movie_name.lower() in df.name.values:
                            st.error("This movie already exists in the database!")
                        else:
                            save_new_movie(new_movie_name, new_movie_description)
                            st.success(f"✅ '{new_movie_name}' has been added to the database!")
                            # Clear cache to reload data
                            st.cache_data.clear()
                            st.cache_resource.clear()
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error adding movie: {str(e)}")
            else:
                st.warning("Please fill in both movie name and description.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------
    # STANDALONE PREDICTION SECTION
    # -------------------------------------------------
    st.markdown("---")
    st.header("🔮 Get Recommendations for Any Movie")
    
    with st.container():
        st.markdown('<div class="predict-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            predict_movie_name_bottom = st.text_input(
                "Enter Movie Name:", 
                placeholder="Type any movie name to get recommendations...",
                key="predict_input_bottom"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            predict_button = st.button("🎯 Get Recommendations", key="predict_movie_bottom")
        
        if predict_button:
            if predict_movie_name_bottom:
                with st.spinner("Finding similar movies..."):
                    recommendations, error = predict_movie_recommendations(predict_movie_name_bottom, df)
                    
                    if error:
                        st.error(error)
                        st.info("💡 **Tip**: Add this movie to the database first using the 'Add New Movie' section above!")
                    elif recommendations:
                        st.subheader(f"🎬 Movies similar to '{predict_movie_name_bottom}':")
                        for rec in recommendations:
                            st.write(f"🎥 **{rec['name']}** (Similarity: {rec['similarity']:.2%})")
                            with st.expander("View Description"):
                                st.write(rec['description'])
                    else:
                        st.info("No recommendations found.")
            else:
                st.warning("Please enter a movie name.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------
    # DATABASE STATISTICS
    # -------------------------------------------------
    
    st.markdown("---")

    with st.container():
        st.markdown('<div class="add-total_movie">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Movies in Database", len(df))
        with col2:
            st.metric("Current User", st.session_state.current_user)
        with col3:
            if st.button("🔄 Refresh Database"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

# -------------------------------------------------
# ROUTING (IMPORTANT FIX)
# -------------------------------------------------
if st.session_state.logged_in:
    movie_page()
else:
    login_page()
