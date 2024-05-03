import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk, Image  # Import Image module from PIL library
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
import os

# Download NLTK resources
nltk.download('vader_lexicon')

class AspectSentimentAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("Aspect-Based Sentiment Analysis")

        # Get current directory
        current_dir = os.path.dirname(__file__)

        # Load background image
        background_path = os.path.join(current_dir, "background.jpg")
        self.background_image = Image.open(background_path)
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        # Create a canvas with background image
        self.canvas = tk.Canvas(master, width=self.background_image.width, height=self.background_image.height)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.background_photo, anchor="nw")

        # Create a frame on the canvas for the content
        self.content_frame = ttk.Frame(self.canvas, style="Custom.TFrame")
        self.content_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Apply some padding to the content frame
        self.content_frame.grid(padx=20, pady=20)

        # Add labels, entry, combobox, and buttons to the content frame
        self.label = ttk.Label(self.content_frame, text="Enter Text:")
        self.label.grid(row=0, column=0, padx=5, pady=5)

        self.text_entry = ttk.Entry(self.content_frame, width=50)
        self.text_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5)

        self.aspect_label = ttk.Label(self.content_frame, text="Select Aspect:")
        self.aspect_label.grid(row=1, column=0, padx=5, pady=5)

        self.aspect_var = tk.StringVar()
        self.aspect_combobox = ttk.Combobox(self.content_frame, textvariable=self.aspect_var,
                                            values=["product", "customer service", "news article",
                                                    "shopping experience"])
        self.aspect_combobox.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        self.analyze_button = ttk.Button(self.content_frame, text="Analyze", command=self.analyze_sentiment)
        self.analyze_button.grid(row=2, column=1, padx=5, pady=10)

        self.quit_button = ttk.Button(self.content_frame, text="Quit", command=master.quit)
        self.quit_button.grid(row=2, column=2, padx=5, pady=10)

        # Fit vectorizer with vocabulary
        self.fit_vectorizer()

        # Load model
        self.model = self.load_model()

        # Initialize VADER Sentiment Intensity Analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()

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

        # Perform sentiment analysis
        sentiment_result = self.aspect_sentiment_analysis(text, aspect)
        sentiment_intensity = self.aspect_sentiment_intensity(text, aspect)
        messagebox.showinfo("Sentiment Analysis Result",
                            f"Sentiment Analysis Result for aspect '{aspect}' in text '{text}': {sentiment_result}\n\n"
                            f"Sentiment Intensity Result for aspect '{aspect}' in text '{text}': {sentiment_intensity}")

    def aspect_sentiment_analysis(self, text, aspect):
        preprocessed_text = self.preprocess_text(text)
        sentiment_scores = self.vader_analyzer.polarity_scores(preprocessed_text)
        sentiment_intensity = sentiment_scores['compound']

        # Classify sentiment based on intensity
        if sentiment_intensity >= 0.05:
            return 'positive'
        elif sentiment_intensity <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def aspect_sentiment_intensity(self, text, aspect):
        preprocessed_text = self.preprocess_text(text)
        sentiment_scores = self.vader_analyzer.polarity_scores(preprocessed_text)
        sentiment_intensity = sentiment_scores['compound']
        return sentiment_intensity

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
