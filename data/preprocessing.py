import os
import random
import numpy as np
import pickle
from collections import defaultdict
from dotenv import load_dotenv
from huggingface_hub import login
from loaders import ItemLoader
import openai

class DatasetProcessor:
    def __init__(self):
        self.dataset_names = [
            "Automotive",
            "Electronics",
            "Office_Products",
            "Tools_and_Home_Improvement",
            "Cell_Phones_and_Accessories",
            "Toys_and_Games",
            "Appliances",
            "Musical_Instruments",
        ]
        self.items = []
        self.slots = defaultdict(list)
        self.sample = []

        # Constants
        self.PRICE_THRESHOLD = 240
        self.MAX_PER_SLOT = 1200
        self.TRAIN_SIZE = 400_000
        self.TEST_SIZE = 2_000

    def setup_environment(self):
        load_dotenv(override=True)
        hf_token = os.getenv("HF_TOKEN")
        openai_key = os.getenv("OPENAI_API_KEY")

        assert hf_token, "HF_TOKEN not found in .env"
        assert openai_key, "OPENAI_API_KEY not found in .env"

        os.environ["HF_TOKEN"] = hf_token
        os.environ["OPENAI_API_KEY"] = openai_key

        openai.api_key = openai_key
        login(hf_token, add_to_git_credential=True)

    def load_items(self):
        print("Loading items from dataset...")
        for name in self.dataset_names:
            loader = ItemLoader(name)
            try:
                self.items.extend(loader.load())
            except Exception as e:
                print(f"Failed to load dataset '{name}': {e}")

    def organize_items_by_price(self):
        print("Organizing items by price...")
        for item in self.items:
            self.slots[round(item.price)].append(item)

    def generate_sample(self):
        print("Generating balanced sample...")
        np.random.seed(42)
        random.seed(42)

        for i in range(1, 1000):
            slot = self.slots[i]
            if not slot:
                continue

            if i >= self.PRICE_THRESHOLD:
                self.sample.extend(slot)
            elif len(slot) <= self.MAX_PER_SLOT:
                self.sample.extend(slot)
            else:
                weights = np.array(
                    [1 if item.category == "Automotive" else 5 for item in slot], dtype=float
                )
                weights /= np.sum(weights)
                selected_indices = np.random.choice(len(slot), size=self.MAX_PER_SLOT, replace=False, p=weights)
                selected = [slot[j] for j in selected_indices]
                self.sample.extend(selected)

        print(f"There are {len(self.sample):,} items in the sample")

    def split_train_test(self):
        print("Splitting sample into train/test sets...")
        random.seed(42)
        random.shuffle(self.sample)
        train = self.sample[:self.TRAIN_SIZE]
        test = self.sample[self.TRAIN_SIZE:self.TRAIN_SIZE + self.TEST_SIZE]
        print(f"Train set: {len(train):,} items")
        print(f"Test set: {len(test):,} items")
        return train, test

    def save_as_pickle(self, data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {filename} with {len(data):,} items.")

    def run(self, save=True):
        self.setup_environment()
        self.load_items()
        self.organize_items_by_price()
        self.generate_sample()
        train, test = self.split_train_test()

        if save:
            self.save_as_pickle(train, "data/train.pkl")
            self.save_as_pickle(test, "data/test.pkl")

        return train, test


if __name__ == "__main__":
    processor = DatasetProcessor()
    train, test = processor.run()
    print(f"Train Example: {train[0]}")
    print(f"Test Example: {test[0]}")