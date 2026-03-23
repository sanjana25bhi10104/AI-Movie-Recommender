

import pandas as pd
from sklearn.neighbors import NearestNeighbors

def run_recommender():
    try:
        df = pd.read_csv('movies.csv')
        
        X = df[['Action', 'Comedy']].values
        movie_titles = df['Movie'].values

        print("--- AI Movie Matcher (KNN) ---")
        print(f"Dataset active: {len(df)} movies indexed.\n")

        print("Rate your preference from 1 to 10:")
        user_act = float(input("Action Level? (1-10): "))
        user_com = float(input("Comedy Level? (1-10): "))
        
        if not (1 <= user_act <= 10 and 1 <= user_com <= 10):
            print("Please keep scores between 1 and 10!")
            return

        model = NearestNeighbors(n_neighbors=3)
        model.fit(X)
        distances, indices = model.kneighbors([[user_act, user_com]])

        print(f"\nAI is calculating the closest matches...")
        print("-" * 35)
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            print(f"{i+1}. {movie_titles[idx]} (Distance: {dist:.2f})")
        print("-" * 35)

    except FileNotFoundError:
        print("Error: 'movies.csv' missing. Please upload it to your repo.")
    except ValueError:
        print("Error: Please enter numbers (e.g., 7.5 or 8) only.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_recommender()
