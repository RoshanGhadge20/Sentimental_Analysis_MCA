import tkinter as tk  # Import the tkinter library for creating the GUI
from tkinter import ttk, messagebox  # Import ttk for themed widgets and messagebox for dialog boxes
from sklearn.feature_extraction.text import CountVectorizer  # Import CountVectorizer for text vectorization
from sklearn.naive_bayes import MultinomialNB  # Import MultinomialNB for Naive Bayes classification
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Import VADER sentiment analyzer from nltk
import nltk  # Import nltk for natural language processing
import re  # Import re for regular expressions
import os  # Import os for operating system functionalities
import spacy  # Import spacy for named entity recognition

# Download the VADER lexicon required for sentiment analysis
nltk.download('vader_lexicon')

class AspectSentimentAnalysisAppwithNER:
    def __init__(self, master):
        self.master = master
        master.title("Aspect-Based Sentiment Analysis")

        # Create a frame for the content
        self.content_frame = ttk.Frame(master)
        self.content_frame.pack(padx=20, pady=20)

        # Add a label for text input
        self.label = ttk.Label(self.content_frame, text="Enter Text:")
        self.label.grid(row=0, column=0, padx=5, pady=5)

        # Add an entry widget for text input
        self.text_entry = ttk.Entry(self.content_frame, width=50)
        self.text_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5)

        # Add a label for aspect selection
        self.aspect_label = ttk.Label(self.content_frame, text="Select Aspect:")
        self.aspect_label.grid(row=1, column=0, padx=5, pady=5)

        # Add a combobox for selecting an aspect
        self.aspect_var = tk.StringVar()
        self.aspect_combobox = ttk.Combobox(self.content_frame, textvariable=self.aspect_var,
                                            values=["product", "customer service", "news article", "shopping experience"])
        self.aspect_combobox.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        # Add a button to trigger sentiment analysis
        self.analyze_button = ttk.Button(self.content_frame, text="Analyze", command=self.analyze_sentiment)
        self.analyze_button.grid(row=2, column=1, padx=5, pady=10)

        # Add a button to quit the application
        self.quit_button = ttk.Button(self.content_frame, text="Quit", command=master.quit)
        self.quit_button.grid(row=2, column=2, padx=5, pady=10)

        # Fit the vectorizer with the vocabulary
        self.fit_vectorizer()

        # Load the sentiment analysis model
        self.model = self.load_model()

        # Initialize VADER Sentiment Intensity Analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Load spaCy NER model
        self.nlp = spacy.load("en_core_web_sm")

    def fit_vectorizer(self):
        data = self.collect_data()
        text_data = [item[0] for item in data]
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(text_data)

    def collect_data(self):
        # Collect training data for the vectorizer
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
        # Preprocess the text by removing special characters and converting to lowercase
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = text.lower()
        return text

    def analyze_sentiment(self):
        # Get text and aspect from the user input
        text = self.text_entry.get()
        aspect = self.aspect_var.get()

        # Show warning if text or aspect is not provided
        if not text:
            messagebox.showwarning("Warning", "Please enter some text.")
            return

        if not aspect:
            messagebox.showwarning("Warning", "Please select an aspect.")
            return

        # Perform sentiment analysis
        sentiment_result = self.aspect_sentiment_analysis(text, aspect)
        sentiment_intensity = self.aspect_sentiment_intensity(text, aspect)

        # Show the sentiment analysis results in a message box
        messagebox.showinfo("Sentiment Analysis Result",
                            f"Sentiment Analysis Result for aspect '{aspect}' in text '{text}': {sentiment_result}\n\n"
                            f"Sentiment Intensity Result for aspect '{aspect}' in text '{text}': {sentiment_intensity}")

    def aspect_sentiment_analysis(self, text, aspect):
        # Preprocess the input text
        preprocessed_text = self.preprocess_text(text)

        # Extract named entities from the text
        entities = self.extract_entities(preprocessed_text)

        # Filter entities relevant to the aspect
        aspect_entities = [entity.text for entity in entities if entity.label_ == "PRODUCT" or entity.label_ == "ORG"]

        # Perform sentiment analysis using VADER
        sentiment = self.vader_analyzer.polarity_scores(preprocessed_text)['compound']

        # Classify sentiment based on intensity and aspect entities
        if sentiment >= 0.05:
            return 'positive' if aspect_entities else 'neutral'
        elif sentiment <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def aspect_sentiment_intensity(self, text, aspect):
        # Preprocess the text
        preprocessed_text = self.preprocess_text(text)

        # Get sentiment intensity scores from VADER
        sentiment_scores = self.vader_analyzer.polarity_scores(preprocessed_text)
        sentiment_intensity = sentiment_scores['compound']
        return sentiment_intensity

    def extract_entities(self, text):
        # Perform named entity recognition using spaCy
        doc = self.nlp(text)
        return doc.ents

    def load_model(self):
        # Load a pretrained sentiment analysis model (currently a placeholder)
        model = MultinomialNB()
        return model


def run_aspect_sentiment_analysis_gui():
    # Create and run the GUI application
    root = tk.Tk()
    app = AspectSentimentAnalysisAppwithNER(root)
    root.mainloop()


if __name__ == "__main__":
    run_aspect_sentiment_analysis_gui()
