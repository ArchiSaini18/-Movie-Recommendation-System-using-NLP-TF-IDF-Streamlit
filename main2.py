import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
from io import BytesIO
import random

# ========================= PAGE CONFIGURATION =========================
st.set_page_config(
    page_title="FilmyX AI - Smart Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= CUSTOM CSS STYLING =========================
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(90deg, #e94560 0%, #533483 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(233, 69, 96, 0.3);
    }
    
    .main-title {
        color: #1a1a2e;
        font-size: 3.5rem;
        font-weight: bold;
        margin: 0;
       text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.3);
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e94560;
    }
    
    /* Movie Card Styles */
    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(233, 69, 96, 0.3);
        border: 1px solid rgba(233, 69, 96, 0.5);
    }
    
    .movie-title {
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .movie-info {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .movie-overview {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        line-height: 1.5;
        margin-top: 0.5rem;
    }
    
    /* Badge Styles */
    .genre-tag {
        background: rgba(233, 69, 96, 0.2);
        color: #e94560;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.75rem;
        border: 1px solid #e94560;
        margin: 2px;
        display: inline-block;
    }
    
    .match-badge {
        background: linear-gradient(90deg, #4ecc71 0%, #2ecc71 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(78, 204, 113, 0.3);
    }
    
    .rating-badge {
        background: linear-gradient(90deg, #f39c12 0%, #f1c40f 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Card Container */
    .card-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Search Section */
    .search-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 3rem;
    }
    
    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background: rgba(0, 0, 0, 0.2);
    }
    
    /* Button Overrides */
    .stButton > button {
        background: linear-gradient(90deg, #e94560 0%, #533483 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(233, 69, 96, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ========================= DATA LOADING =========================

@st.cache_data
def load_movie_data():
    """Load comprehensive movie dataset with enhanced features"""
    movies = pd.DataFrame({
        'movie_id': range(1, 41),
        'title': [
            "The Shawshank Redemption", "The Godfather", "The Dark Knight", "Pulp Fiction", 
            "Forrest Gump", "Inception", "The Matrix", "Goodfellas", "Interstellar", 
            "The Silence of the Lambs", "Star Wars: The Empire Strikes Back", 
            "The Lord of the Rings: The Return of the King", "Fight Club", "The Departed", 
            "Gladiator", "Titanic", "Jurassic Park", "Avatar", "The Avengers", 
            "Django Unchained", "The Prestige", "Memento", "The Green Mile", 
            "Saving Private Ryan", "The Lion King", "Toy Story", "Finding Nemo", 
            "WALL-E", "Up", "Inside Out", "Parasite", "Joker", "1917", "Dunkirk",
            "Mad Max: Fury Road", "Blade Runner 2049", "The Social Network", 
            "Whiplash", "La La Land", "Arrival"
        ],
        'genres': [
            "Drama, Crime", "Crime, Drama", "Action, Crime, Drama", "Crime, Drama", 
            "Drama, Romance", "Action, Sci-Fi, Thriller", "Action, Sci-Fi", "Crime, Drama", 
            "Sci-Fi, Drama, Adventure", "Crime, Drama, Thriller", "Action, Adventure, Sci-Fi", 
            "Adventure, Fantasy, Action", "Drama", "Crime, Drama, Thriller", 
            "Action, Drama, Adventure", "Romance, Drama", "Adventure, Sci-Fi, Thriller", 
            "Action, Adventure, Sci-Fi", "Action, Adventure, Sci-Fi", "Western, Drama", 
            "Drama, Mystery, Thriller", "Mystery, Thriller", "Crime, Drama, Fantasy", 
            "Drama, War", "Animation, Adventure, Drama", "Animation, Adventure, Comedy", 
            "Animation, Adventure, Comedy", "Animation, Adventure, Sci-Fi", 
            "Animation, Adventure, Comedy", "Animation, Adventure, Comedy", 
            "Drama, Thriller", "Crime, Drama, Thriller", "Drama, War", "Action, Drama, War",
            "Action, Adventure, Sci-Fi", "Sci-Fi, Drama, Mystery", "Biography, Drama", 
            "Drama, Music", "Comedy, Drama, Music, Romance", "Sci-Fi, Drama, Mystery"
        ],
        'director': [
            "Frank Darabont", "Francis Ford Coppola", "Christopher Nolan", "Quentin Tarantino",
            "Robert Zemeckis", "Christopher Nolan", "Lana Wachowski", "Martin Scorsese",
            "Christopher Nolan", "Jonathan Demme", "Irvin Kershner", "Peter Jackson",
            "David Fincher", "Martin Scorsese", "Ridley Scott", "James Cameron",
            "Steven Spielberg", "James Cameron", "Joss Whedon", "Quentin Tarantino",
            "Christopher Nolan", "Christopher Nolan", "Frank Darabont", "Steven Spielberg",
            "Roger Allers", "John Lasseter", "Andrew Stanton", "Andrew Stanton",
            "Pete Docter", "Pete Docter", "Bong Joon-ho", "Todd Phillips", 
            "Sam Mendes", "Christopher Nolan", "George Miller", "Denis Villeneuve",
            "David Fincher", "Damien Chazelle", "Damien Chazelle", "Denis Villeneuve"
        ],
        'cast': [
            "Tim Robbins, Morgan Freeman", "Marlon Brando, Al Pacino", "Christian Bale, Heath Ledger",
            "John Travolta, Samuel L. Jackson", "Tom Hanks, Robin Wright", "Leonardo DiCaprio, Marion Cotillard",
            "Keanu Reeves, Laurence Fishburne", "Robert De Niro, Ray Liotta", "Matthew McConaughey, Anne Hathaway",
            "Jodie Foster, Anthony Hopkins", "Mark Hamill, Harrison Ford", "Elijah Wood, Viggo Mortensen",
            "Brad Pitt, Edward Norton", "Leonardo DiCaprio, Matt Damon", "Russell Crowe, Joaquin Phoenix",
            "Leonardo DiCaprio, Kate Winslet", "Sam Neill, Laura Dern", "Sam Worthington, Zoe Saldana",
            "Robert Downey Jr., Chris Evans", "Jamie Foxx, Christoph Waltz", "Christian Bale, Hugh Jackman",
            "Guy Pearce, Carrie-Anne Moss", "Tom Hanks, Michael Clarke Duncan", "Tom Hanks, Matt Damon",
            "Matthew Broderick, James Earl Jones", "Tom Hanks, Tim Allen", "Albert Brooks, Ellen DeGeneres",
            "Ben Burtt, Elissa Knight", "Ed Asner, Jordan Nagai", "Amy Poehler, Phyllis Smith",
            "Song Kang-ho, Lee Sun-kyun", "Joaquin Phoenix, Robert De Niro", "George MacKay, Dean-Charles Chapman",
            "Fionn Whitehead, Tom Hardy", "Tom Hardy, Charlize Theron", "Ryan Gosling, Harrison Ford",
            "Jesse Eisenberg, Andrew Garfield", "Miles Teller, J.K. Simmons", "Ryan Gosling, Emma Stone",
            "Amy Adams, Jeremy Renner"
        ],
        'overview': [
            "Two imprisoned men bond over years finding solace and redemption through acts of common decency",
            "The aging patriarch of an organized crime dynasty transfers control to his reluctant son",
            "Batman must accept one of the greatest psychological tests to fight injustice and chaos in Gotham",
            "Various interconnected stories of crime in Los Angeles told in non-linear fashion",
            "The presidencies of Kennedy and Johnson unfold through the perspective of an Alabama man",
            "A thief who steals corporate secrets through dream-sharing technology is given a final job",
            "A computer hacker learns about the true nature of his reality and his role in the war against controllers",
            "The story of Henry Hill and his life in the mob covering three decades of crime",
            "A team of explorers travel through a wormhole in space attempting to ensure humanity's survival",
            "A young FBI cadet must receive help from an incarcerated cannibalistic killer to catch another serial killer",
            "After the Rebels are overpowered by the Empire Luke Skywalker begins Jedi training with Yoda",
            "A meek Hobbit and his companions embark on a journey to destroy a powerful ring and save Middle-earth",
            "An insomniac office worker forms an underground fight club that evolves into something more",
            "An undercover cop and a mole in the police try to identify each other while infiltrating an Irish gang",
            "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family",
            "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious ill-fated ship",
            "Scientists clone dinosaurs to create a theme park but things go wrong when the creatures break free",
            "A paraplegic Marine becomes part of the Na'vi world on Pandora and must choose sides in a conflict",
            "Earth's mightiest heroes must come together and learn to fight as a team to stop an alien invasion",
            "A freed slave sets out to rescue his wife from a brutal Mississippi plantation owner with a bounty hunter",
            "Two magicians engage in a competitive rivalry that turns dangerous and tests the limits of their craft",
            "A man with short-term memory loss attempts to track down his wife's murderer using notes and tattoos",
            "The lives of guards on Death Row are affected by one of their charges a man with a mysterious gift",
            "Following the Normandy Landings a group of soldiers goes behind enemy lines to find a paratrooper",
            "Lion prince Simba flees his kingdom only to learn the true meaning of responsibility and bravery",
            "A cowboy doll is profoundly threatened when a new spaceman action figure supplants him as top toy",
            "A clownfish searches for his captured son across the ocean with the help of a forgetful fish",
            "A robot left on Earth falls in love and embarks on a space journey that will decide the fate of mankind",
            "An elderly man recalls his greatest adventure with his late wife through a young stowaway on his airborne house",
            "The emotions inside a young girl's mind struggle to help her adapt to a big life change",
            "A poor family schemes to become employed by a wealthy family and infiltrate their household",
            "A mentally troubled comedian descends into insanity and becomes a criminal mastermind in Gotham",
            "Two British soldiers must cross enemy territory to deliver a message that could save 1600 men",
            "Allied soldiers are surrounded by enemy forces and must evacuate from Dunkirk beach during World War II",
            "A post-apocalyptic warlord chases down a rebel who has escaped in the company of his five wives",
            "A young blade runner discovers a secret that could plunge society into chaos and finds a former runner",
            "The story of how Mark Zuckerberg created Facebook while dealing with lawsuits and broken friendships",
            "A promising young drummer enrolls at a music conservatory and faces a terrifying instructor",
            "An aspiring actress and a jazz musician fall in love while pursuing their dreams in Los Angeles",
            "A linguist is recruited by the military to communicate with aliens who have arrived on Earth"
        ],
        'year': [
            1994, 1972, 2008, 1994, 1994, 2010, 1999, 1990, 2014, 1991, 1980, 2003, 1999, 2006,
            2000, 1997, 1993, 2009, 2012, 2012, 2006, 2000, 1999, 1998, 1994, 1995, 2003, 2008,
            2009, 2015, 2019, 2019, 2019, 2017, 2015, 2017, 2010, 2014, 2016, 2016
        ],
        'rating': [
            9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.6, 8.7, 8.9, 8.8, 8.5,
            8.5, 7.9, 8.2, 7.9, 8.0, 8.4, 8.5, 8.4, 8.6, 8.6, 8.5, 8.3, 8.2, 8.4,
            8.3, 8.1, 8.5, 8.4, 8.2, 7.8, 8.1, 8.0, 7.8, 8.5, 8.0, 7.9
        ],
        'poster_url': [
            "https://image.tmdb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
            "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
            "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
            "https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
            "https://image.tmdb.org/t/p/w500/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg",
            "https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg",
            "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
            "https://image.tmdb.org/t/p/w500/aKuFiU82s5ISJpGZp7YkIr3kCUd.jpg",
            "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
            "https://image.tmdb.org/t/p/w500/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg",
            "https://image.tmdb.org/t/p/w500/2l05cFWJacyIsTpsqSgH0wQXe4V.jpg",
            "https://image.tmdb.org/t/p/w500/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg",
            "https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
            "https://image.tmdb.org/t/p/w500/nT97ifVT2J1yMQmeq20Qblg61T.jpg",
            "https://image.tmdb.org/t/p/w500/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg",
            "https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg",
            "https://image.tmdb.org/t/p/w500/b1xCNnyrPebIc7EWNZIa6jhb1Ww.jpg",
            "https://image.tmdb.org/t/p/w500/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg",
            "https://image.tmdb.org/t/p/w500/RYMX2wcKCBAr24UyPD7xwmjaTn.jpg",
            "https://image.tmdb.org/t/p/w500/7oWY8VDWW7thTzWh3OKYRkWUlD5.jpg",
            "https://image.tmdb.org/t/p/w500/bdN3gXuIZYaJP7ftKK2sU0nPtEA.jpg",
            "https://image.tmdb.org/t/p/w500/yuNs09hvpHVU1cBTCAk9zxsL2oW.jpg",
            "https://image.tmdb.org/t/p/w500/velWPhVMQeQKcxggNEU8YmIo52R.jpg",
            "https://image.tmdb.org/t/p/w500/uqx37vgn6qXU6Z78QY4FKw8c3Fv.jpg",
            "https://image.tmdb.org/t/p/w500/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg",
            "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
            "https://image.tmdb.org/t/p/w500/eHuGQ10FUzK1mdOY69wF5pGgEf5.jpg",
            "https://image.tmdb.org/t/p/w500/hbhFnRzzg6ZDmm8YAmxBnQpQIPh.jpg",
            "https://image.tmdb.org/t/p/w500/nk11pvocdb5zbFhX5oq5YiLPYMo.jpg",
            "https://image.tmdb.org/t/p/w500/2H1TmgdfNtsKlU9jKdeNyYL5y8T.jpg",
            "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
            "https://image.tmdb.org/t/p/w500/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg",
            "https://image.tmdb.org/t/p/w500/iZf0KyrE25z1sage4SYFLCCrMi9.jpg",
            "https://image.tmdb.org/t/p/w500/f4AFVKFhZGdDbCxCBrHPvvqsjN3.jpg",
            "https://image.tmdb.org/t/p/w500/hA2ple9q4qnwxp3hKVNhroipsir.jpg",
            "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg",
            "https://image.tmdb.org/t/p/w500/n0ybibhJtQ5icDqTp8eRytcIHJx.jpg",
            "https://image.tmdb.org/t/p/w500/lIv1QinFqz4dlp5U4lQ6HaiskOZ.jpg",
            "https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg",
            "https://image.tmdb.org/t/p/w500/hLudzvGfpi6rCWzqbcui69KmIMi.jpg"
        ]
    })
    
    # Create comprehensive feature text combining multiple attributes
    movies['features'] = (
        movies['genres'] + ' ' + 
        movies['director'] + ' ' + 
        movies['cast'] + ' ' + 
        movies['overview']
    )
    
    return movies

@st.cache_resource
def create_similarity_matrix(movies):
    """Create TF-IDF matrix and calculate cosine similarity"""
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )
    
    tfidf_matrix = tfidf.fit_transform(movies['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

# ========================= RECOMMENDATION ENGINE =========================

def get_recommendations(movie_title, movies, cosine_sim, filters, n_recommendations=8):
    """Get movie recommendations with filtering"""
    # Get the index of the movie
    idx = movies[movies['title'] == movie_title].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Apply filters
    filtered_recommendations = []
    for movie_idx, score in sim_scores[1:]:  # Skip the movie itself
        movie = movies.iloc[movie_idx]
        
        # Apply year filter
        if movie['year'] < filters['year_range'][0] or movie['year'] > filters['year_range'][1]:
            continue
        
        # Apply rating filter
        if movie['rating'] < filters['min_rating']:
            continue
        
        # Apply genre filter if selected
        if filters['selected_genres']:
            movie_genres = set(movie['genres'].split(', '))
            if not any(genre in movie_genres for genre in filters['selected_genres']):
                continue
        
        filtered_recommendations.append((movie_idx, score))
        
        if len(filtered_recommendations) >= n_recommendations:
            break
    
    # Create recommendations dataframe
    if filtered_recommendations:
        movie_indices = [i[0] for i in filtered_recommendations]
        similarity_scores = [i[1] for i in filtered_recommendations]
        
        recommendations = movies.iloc[movie_indices].copy()
        recommendations['similarity'] = similarity_scores
        return recommendations
    else:
        return pd.DataFrame()

# ========================= UI COMPONENTS =========================

def display_movie_card(movie, show_similarity=False, compact=False):
    """Display movie card with poster and information"""
    if compact:
        # Compact view for grid display
        st.markdown(f"""
        <div class="movie-card">
            <img src="{movie['poster_url']}" style="width:100%; border-radius:10px; margin-bottom:10px;">
            <div class="movie-title">{movie['title']}</div>
            <div class="movie-info">⭐ {movie['rating']} | 📅 {movie['year']}</div>
        """, unsafe_allow_html=True)
        
        if show_similarity and 'similarity' in movie:
            match_percent = int(movie['similarity'] * 100)
            st.markdown(f'<span class="match-badge">{match_percent}% Match</span>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Detailed view
        col1, col2 = st.columns([1, 2])
        
        with col1:
            try:
                response = requests.get(movie['poster_url'], timeout=5)
                img = Image.open(BytesIO(response.content))
                st.image(img, use_container_width=True)
            except:
                st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
        
        with col2:
            st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)
            
            # Display director and cast
            st.markdown(f'<div class="movie-info">🎬 Director: {movie["director"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="movie-info">🎭 Cast: {movie["cast"]}</div>', unsafe_allow_html=True)
            
            # Display genres
            genres = movie['genres'].split(', ')
            genre_html = ' '.join([f'<span class="genre-tag">{g}</span>' for g in genres])
            st.markdown(genre_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display rating, year, and match
            rating_html = f'<span class="rating-badge">⭐ {movie["rating"]}</span>'
            year_html = f'<span style="color: rgba(255,255,255,0.7); margin-left: 10px;">📅 {movie["year"]}</span>'
            
            if show_similarity and 'similarity' in movie:
                match_percent = int(movie['similarity'] * 100)
                match_html = f'<span class="match-badge" style="margin-left: 10px;">{match_percent}% Match</span>'
                st.markdown(rating_html + year_html + match_html, unsafe_allow_html=True)
            else:
                st.markdown(rating_html + year_html, unsafe_allow_html=True)
            
            # Display overview
            st.markdown(f'<div class="movie-overview">{movie["overview"]}</div>', unsafe_allow_html=True)

# ========================= MAIN APPLICATION =========================

def main():
    # Load data
    movies = load_movie_data()
    cosine_sim = create_similarity_matrix(movies)
    
    # Extract all unique genres
    all_genres = set()
    for genres_str in movies['genres']:
        all_genres.update(genres_str.split(', '))
    all_genres = sorted(list(all_genres))
    
    # ========================= SIDEBAR FILTERS =========================
    st.sidebar.markdown("## 🎯 Discovery Filters")
    
    year_range = st.sidebar.slider(
        "Release Year",
        int(movies['year'].min()),
        int(movies['year'].max()),
        (int(movies['year'].min()), int(movies['year'].max()))
    )
    
    min_rating = st.sidebar.slider(
        "Minimum Rating ⭐",
        0.0,
        10.0,
        7.5,
        0.1
    )
    
    selected_genres = st.sidebar.multiselect(
        "Filter by Genre",
        all_genres,
        default=[]
    )
    
    st.sidebar.markdown("---")
    
    # Random movie button
    if st.sidebar.button("🎲 Surprise Me!", use_container_width=True):
        filtered_movies = movies[
            (movies['year'] >= year_range[0]) & 
            (movies['year'] <= year_range[1]) & 
            (movies['rating'] >= min_rating)
        ]
        
        if len(filtered_movies) > 0:
            random_movie = filtered_movies.sample(1).iloc[0]
            st.session_state.selected_movie = random_movie['title']
            st.session_state.show_recommendations = True
            st.rerun()
    
    # Reset filters button
    if st.sidebar.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📊 About This App
    
    **FilmyX AI** uses advanced machine learning algorithms to recommend movies based on:
    
    - 🎬 **Content Similarity**: Genre, plot, themes
    - 🎭 **Cast & Crew**: Directors and actors
    - 📝 **Story Analysis**: Plot descriptions
    
    **Algorithm**: TF-IDF Vectorization + Cosine Similarity
    
    ---
    
    Made with ❤️ using Streamlit
    """)
    
    # ========================= MAIN HEADER =========================
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎬 FilmyX AI</h1>
        <p class="subtitle">Discover Your Next Favorite Movie with Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================= INITIALIZE SESSION STATE =========================
    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = None
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    
    # ========================= SEARCH SECTION =========================
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: white; margin-top: 0;">🔍 Find Similar Movies</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        selected_movie = st.selectbox(
            "Select a movie you enjoyed:",
            movies['title'].tolist(),
            index=None,
            placeholder="Choose a movie..."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎯 Get Recommendations", use_container_width=True):
            if selected_movie:
                st.session_state.selected_movie = selected_movie
                st.session_state.show_recommendations = True
                st.rerun()
            else:
                st.warning("Please select a movie first!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================= RECOMMENDATIONS DISPLAY =========================
    if st.session_state.show_recommendations and st.session_state.selected_movie:
        
        # Display selected movie
        st.markdown('<div class="section-header">📽️ Your Selected Movie</div>', unsafe_allow_html=True)
        
        selected_movie_data = movies[movies['title'] == st.session_state.selected_movie].iloc[0]
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        display_movie_card(selected_movie_data, show_similarity=False, compact=False)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Get recommendations
        st.markdown('<div class="section-header">✨ Recommended Movies for You</div>', unsafe_allow_html=True)
        
        filters = {
            'year_range': year_range,
            'min_rating': min_rating,
            'selected_genres': selected_genres
        }
        
        with st.spinner("🎬 Analyzing movie features and finding perfect matches..."):
            recommendations = get_recommendations(
                st.session_state.selected_movie,
                movies,
                cosine_sim,
                filters,
                n_recommendations=8
            )
        
        if len(recommendations) > 0:
            # Display recommendations in 2 columns
            cols = st.columns(2)
            
            for idx, (_, movie) in enumerate(recommendations.iterrows()):
                with cols[idx % 2]:
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    display_movie_card(movie, show_similarity=True, compact=False)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ No recommendations found with current filters. Try adjusting the filters in the sidebar.")
    
    else:
        # ========================= POPULAR MOVIES SECTION =========================
        st.markdown('<div class="section-header">🔥 Popular & Highly Rated Movies</div>', unsafe_allow_html=True)
        st.markdown('<p style="color: rgba(255,255,255,0.7); margin-bottom: 2rem;">Discover some of the highest-rated movies of all time</p>', unsafe_allow_html=True)
        
        top_movies = movies.nlargest(9, 'rating')
        
        cols = st.columns(3)
        for idx, (_, movie) in enumerate(top_movies.iterrows()):
            with cols[idx % 3]:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                display_movie_card(movie, show_similarity=False, compact=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================= FOOTER =========================
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3 style="color: white; margin-bottom: 1rem;">🎬 FilmyX AI - Smart Movie Recommendations</h3>
        <p><strong>Technology Stack:</strong></p>
        <p>🧠 <strong>ML Algorithm:</strong> TF-IDF Vectorization + Cosine Similarity (Scikit-learn)</p>
        <p>🎨 <strong>Frontend:</strong> Streamlit with Custom CSS</p>
        <p>📊 <strong>Features:</strong> Content-Based Filtering | Advanced Search Filters | Smart Recommendations</p>
        <br>
        <p>Built with ❤️ using Python, Streamlit, and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
