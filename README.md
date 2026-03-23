# AI Movie Recommender
BYOP(Bring Your Own Project) Submission

Course: Fundamentals in AI and ML


This is a simple CLI (Command Line) tool that uses the KNN algorithm to recommend a movie based on your favorite genres.


How to Run:-

Install dependencies: pip install -r requirements.txt

Run the script: python main.py

Important Setup Note: The main.py script and the movies.csv file must be kept in the same directory. The program uses a relative path to load the 60+ movie dataset. If the files are separated, you will receive a FileNotFoundError. If you are using Google Colab to test this, please upload the movies.csv to the session storage before running the script.


How it Works:-

Algorithm: K-Nearest Neighbors (KNN).

Logic: The program takes your input (1-10 for Action and Comedy) and finds the movie in the database with the "closest" mathematical distance to your choice.

No GUI: This project runs entirely in the terminal as per the submission rules.


Files:-

main.py: The core Python code.

movies.csv: Contains a curated dataset of 60+ Bollywood hits (2021-2026) for accurate genre matching.

requirements.txt: Required libraries (scikit-learn, numpy).

README.md: This guide.
