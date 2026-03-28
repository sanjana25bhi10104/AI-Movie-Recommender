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
    print("-" * 50)
    print("🎬 UNIVERSAL BOLLYWOOD AI RECOMMENDER")
    print("-" * 50)

    df = get_data()
    if df is None: return

    X = df[['Action', 'Comedy']].values
    names = df['Movie'].values

    try:
        act = float(input("\nEnter Action Level (1-10): "))
        com = float(input("Enter Comedy Level (1-10): "))

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
