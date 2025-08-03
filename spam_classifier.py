# spam_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

def train_and_evaluate_model():
    """
    This function loads the dataset, trains a spam classifier,
    and evaluates its performance.
    """
    # --- 1. Load and Prepare the Dataset ---
    try:
        # The dataset is expected to be a tab-separated file with two columns: 'label' and 'message'
        df = pd.read_csv('spam.csv', sep='\t', names=['label', 'message'], encoding='latin-1')
    except FileNotFoundError:
        print("Error: 'spam.csv' not found.")
        print("Please download the dataset from https://archive.ics.uci.edu/ml/datasets/sms+spam+collection")
        print("And place 'SMSSpamCollection' as 'spam.csv' in the project directory.")
        return

    # Map labels to numerical values for the model
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # --- 2. Split Data into Training and Testing sets ---
    # We use 80% of the data for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], 
        df['label'], 
        test_size=0.2, 
        random_state=42 # for reproducibility
    )

    # --- 3. Feature Extraction (TF-IDF) ---
    # Convert text data into numerical vectors using Term Frequency-Inverse Document Frequency
    # This helps the model understand which words are important.
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # --- 4. Train the Classification Model ---
    # We use the Multinomial Naive Bayes classifier, which is well-suited for text classification
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # --- 5. Make Predictions ---
    y_pred = model.predict(X_test_tfidf)

    # --- 6. Evaluate the Model's Performance ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # --- 7. Display Results ---
    print("--- Spam Classifier Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (How many selected items are relevant?)")
    print(f"Recall:    {recall:.4f} (How many relevant items are selected?)")
    print(f"F1 Score:  {f1:.4f} (A balance between Precision and Recall)")
    print("\n--- Confusion Matrix ---")
    print("         Predicted Ham | Predicted Spam")
    print(f"Actual Ham:   {conf_matrix[0][0]:<12} | {conf_matrix[0][1]}")
    print(f"Actual Spam:  {conf_matrix[1][0]:<12} | {conf_matrix[1][1]}")
    print("--------------------------------------")


if __name__ == '__main__':
    train_and_evaluate_model()
