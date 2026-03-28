import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def load_data():
    """
    Locates and loads movies.csv regardless of where the script is executed.
    This prevents 'FileNotFoundError' during evaluation.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'movies.csv')
    
    if not os.path.exists(csv_path):
        print(f"\n[SYSTEM ERROR] Could not find 'movies.csv' at: {csv_path}")
        print("Please ensure the CSV file is in the same folder as this script.")
        return None
    
    return pd.read_csv(csv_path)

def run_recommender():
    print("="*50)
    print("🎬 BOLLYWOOD AI MOVIE MATCHER (2021-2026)")
    print("="*50)

    movie_db = load_data()
    if movie_db is None:
        return

    features = movie_db[['Action', 'Comedy']].values
    titles = movie_db['Movie'].values

    try:
        print("\nEnter your preferences (Scale: 1.0 to 10.0)")
        user_action = float(input("Desired Action Level: "))
        user_comedy = float(input("Desired Comedy Level: "))

        if not (0 <= user_action <= 10 and 0 <= user_comedy <= 10):
            print("\n[!] Please keep scores between 1 and 10 for accurate results.")
            return

    except ValueError:
        print("\n[!] Invalid input detected. Please enter numerical values only.")
        return

    model = NearestNeighbors(n_neighbors=10, metric='euclidean')
    model.fit(features)
    
    distances, indices = model.kneighbors([[user_action, user_comedy]])

    print(f"\nTop 10 Matches for Action({user_action}) & Comedy({user_comedy}):")
    print("-" * 55)
    print(f"{'#':<3} | {'Movie Title':<35} | {'Match'}")
    print("-" * 55)

    for i in range(len(indices[0])):
        idx = indices[0][i]
        dist = distances[0][i]
        similarity = max(0, 100 - (dist * 10))
        print(f"{i+1:<3} | {titles[idx]:<35} | {similarity:.1f}%")
    
    print("-" * 55)
    print("Recommendation search complete.\n")

if __name__ == "__main__":
    run_recommender()
