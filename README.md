# AI Movie Recommender
BYOP(Bring Your Own Project) Submission

Course: Fundamentals in AI and ML


This is a simple CLI-based Machine Learning application that uses the K-Nearest Neighbors (KNN) algorithm to recommend the top 10 Bollywood movies (2021–2026) based on user-defined Action and Comedy intensity.


How to Run:-

Install dependencies: pip install -r requirements.txt

Run the script: python main.py

Project Structure
main.py: Core logic, input validation, and KNN implementation.

movies.csv: Dataset containing 150+ categorized Bollywood films.

README.md: System documentation and setup guide.

Note: Ensure movies.csv is in the same folder as main.py before execution.



How it Works:-

Algorithm: K-Nearest Neighbors (KNN).

Logic: The system treats movie genres as Feature Vectors on a 2D plane. When a user inputs their preference, the model calculates the Euclidean Distance between the input and the dataset to identify the 10 closest "neighbors."

No GUI: This project runs entirely in the terminal as per the submission rules.


Requirements & Setup:-

Language: Python 3.8+

Libraries: pandas, scikit-learn, numpy

Environment: Works on Local Terminal, VS Code, or Google Colab.



