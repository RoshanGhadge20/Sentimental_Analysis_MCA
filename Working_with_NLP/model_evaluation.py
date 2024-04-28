import numpy as np
from sklearn.naive_bayes import MultinomialNB


def evaluate_model(model, train_features, train_labels, test_features, test_labels):
    # Train the model
    model.fit(train_features, train_labels)
    # Evaluate the performance of the trained model
    accuracy = model.score(test_features, test_labels)
    return accuracy


if __name__ == "__main__":
    train_features = np.array([[1, 0, 1], [0, 1, 0]])  # Features extracted from training text data
    train_labels = np.array([1, 0])  # Corresponding sentiment labels for training data
    test_features = np.array([[0, 1, 0], [1, 0, 1]])  # Features extracted from test text data
    test_labels = np.array([0, 1])  # Corresponding sentiment labels for test data

    evaluation_results = evaluate_model(MultinomialNB(), train_features, train_labels, test_features, test_labels)
    print("Evaluation results:", evaluation_results)