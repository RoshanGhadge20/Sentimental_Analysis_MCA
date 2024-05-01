import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.exceptions import NotFittedError
import numpy as np
import re


class AspectSentimentAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("Aspect-Based Sentiment Analysis")

        self.label = ttk.Label(master, text="Enter Text:")
        self.label.grid(row=0, column=0)

        self.text_entry = ttk.Entry(master, width=50)
        self.text_entry.grid(row=0, column=1, columnspan=2)

        self.aspect_label = ttk.Label(master, text="Select Aspect:")
        self.aspect_label.grid(row=1, column=0)

        self.aspect_var = tk.StringVar()
        self.aspect_combobox = ttk.Combobox(master, textvariable=self.aspect_var,
                                            values=["product", "customer service", "news article",
                                                    "shopping experience"])
        self.aspect_combobox.grid(row=1, column=1, columnspan=2)

        self.analyze_button = ttk.Button(master, text="Analyze", command=self.analyze_sentiment)
        self.analyze_button.grid(row=2, column=1)

        self.quit_button = ttk.Button(master, text="Quit", command=master.quit)
        self.quit_button.grid(row=2, column=2)

        # Fit vectorizer with vocabulary
        self.fit_vectorizer()

        # Load model
        self.model = self.load_model()

    def fit_vectorizer(self):
        data = self.collect_data()
        text_data = [item[0] for item in data]
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(text_data)

    def collect_data(self):
        data = [
            ("This product is amazing! I love it.", "positive"),
            ("Terrible experience with customer service.", "negative"),
            ("The news article provided great insights.", "positive"),
            ("I had a pleasant experience shopping with this company.", "positive"),
            ("Its very bad product, i have disliked it.", "negative"),
            ("I am not agree with you.", "negative")
        ]
        return data

    def preprocess_text(self, text):
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
        text = text.lower()  # Convert text to lowercase
        return text

    def analyze_sentiment(self):
        text = self.text_entry.get()
        aspect = self.aspect_var.get()

        if not text:
            messagebox.showwarning("Warning", "Please enter some text.")
            return

        if not aspect:
            messagebox.showwarning("Warning", "Please select an aspect.")
            return

        # Transform text using the fitted vectorizer
        features = self.vectorizer.transform([text])

        # Perform sentiment analysis
        sentiment_result = self.aspect_sentiment_analysis(features, text, aspect)
        print(sentiment_result)

        messagebox.showinfo("Sentiment Analysis Result",
                            f"Sentiment Analysis Result for aspect '{aspect}' in text '{text}': {sentiment_result}")

    def aspect_sentiment_analysis(self, features, text, aspect):
        preprocessed_text = self.preprocess_text(text)
        features = self.vectorizer.transform([preprocessed_text])

        try:
            sentiment_result = self.model.predict(features)[0]
            return sentiment_result
        except NotFittedError:
            messagebox.showwarning("Warning", "Please fit the model first.")
            return None
    def load_model(self):
        model = MultinomialNB()
        # You can load the trained model here
        return model


def run_aspect_sentiment_analysis_gui():
    root = tk.Tk()
    app = AspectSentimentAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_aspect_sentiment_analysis_gui()
