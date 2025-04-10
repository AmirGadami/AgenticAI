import os
import re
from typing import List
from sentence_transformers import SentenceTransformer
import joblib
from agents.agent import Agent



class RandomForestAgent(Agent):
    name = "Random Forest Agent"
    color = Agent.MAGENTA

    def __init__(self):


        self.log("Random Forest Agent is initializing")
        self.vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.model = joblib.load('random_forest_model.pkl')
        self.log("Random Forest Agent is ready")


    def price(self, description: str) -> float:
        self.log("Random Forest Agent is starting a prediction")
        vectore = self.vectorizer.encode([description])
        self.log("Random Forest Agent is starting a prediction")
        vector = self.vectorizer.encode([description])
        result = max(0, self.model.predict(vector)[0])
        self.log(f"Random Forest Agent completed - predicting ${result:.2f}")
        return result

    