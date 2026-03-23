

import pandas as pd
from sklearn.neighbors import NearestNeighbors

def run_bollywood_recommender():
    try:
        df = pd.read_csv('movies.csv')
        X = df[['Action', 'Comedy']].values
        movie_titles = df['Movie'].values

        print("--- Bollywood AI Matcher (2021-2026) ---")
        print(f"Total Movies in Database: {len(df)}")
        print("Finding your perfect Desi match...\n")

        print("Rate what you want to watch (1 = Low, 10 = High):")
        user_act = float(input("Action/Masala Level? : "))
        user_com = float(input("Comedy/Humor Level?  : "))
        
        if not (0 <= user_act <= 10 and 0 <= user_com <= 10):
            print("Please enter a value between 1 and 10!")
            return

        model = NearestNeighbors(n_neighbors=3)
        model.fit(X)
        distances, indices = model.kneighbors([[user_act, user_com]])

        print(f"\nAI Recommendations for you:")
        print("=" * 35)
        for i in range(len(indices[0])):
            idx = indices[0][i]
            name = movie_titles[idx]
            match_percent = max(0, 100 - (distances[0][i] * 10))
            print(f"{i+1}. {name} ({match_percent:.1f}% Match)")
        print("=" * 35)

    except FileNotFoundError:
        print("Error: 'movies.csv' not found. Upload it to GitHub first!")
    except ValueError:
        print("Error: Please enter a number (e.g. 7 or 8.5).")

if __name__ == "__main__":
    run_bollywood_recommender()
