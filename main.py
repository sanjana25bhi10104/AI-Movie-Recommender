import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_data():
    """Universal path handler for Local, VS Code, and Colab."""
    filename = 'movies.csv'
    base_path = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    full_path = os.path.join(base_path, filename)

    if not os.path.exists(full_path):
        print(f"\n[!] ERROR: '{filename}' not found.")
        print(f"Please ensure the CSV is in: {base_path}")
        return None
    return pd.read_csv(full_path)

def start_app():
    movie_db = get_data() 
    if movie_db is None: return
    total_movies = len(movie_db)
    print("-" * 50)
    print("AI-MOVIE-RECOMMENDER")
    print("🍿 Desi Movie Matcher: Your 2021-2026 Watchlist")
    print(f"Total Movies in Database: {total_movies}")
    print("Finding your perfect Desi match based on vibes...")
    print("-" * 50)

    df = get_data()
    if df is None: return

    X = df[['Action', 'Comedy']].values
    names = df['Movie'].values

    try:
        print("\nRate your mood (1 = None, 10 = Max Masala):")
        act = float(input("How much Action/Masala are you craving? : "))
        com = float(input("How much Comedy/Humor are you craving?  : "))

        if not (0 <= act <= 10 and 0 <= com <= 10):
            print("Please enter values between 1 and 10.")
            return

        engine = NearestNeighbors(n_neighbors=10, metric='euclidean')
        engine.fit(X)
        dist, idx = engine.kneighbors([[act, com]])

        print(f"\nTop 10 Recommendations:")
        print("=" * 55)
        for i in range(len(idx[0])):
            m_idx = idx[0][i]
            d_val = dist[0][i]
            score = max(0, 100 - (d_val * 10))
            print(f"{i+1:<3} | {names[m_idx]:<35} | {score:.1f}% Match")
        print("=" * 55)

    except ValueError:
        print("[!] Error: Please enter numbers only (e.g., 7.5).")

if __name__ == "__main__":
    start_app()
