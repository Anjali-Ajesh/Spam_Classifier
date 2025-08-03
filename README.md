# Spam_Classifier
A machine learning project in Python that classifies SMS messages as either "spam" or "ham" (legitimate). This script uses the scikit-learn library to build and evaluate a text classification model.

## Features

-   **Data Preprocessing:** Cleans and prepares text data for machine learning.
-   **TF-IDF Vectorization:** Converts text messages into meaningful numerical feature vectors.
-   **Naive Bayes Classifier:** Uses a Multinomial Naive Bayes algorithm, a highly effective model for text classification tasks.
-   **Model Evaluation:** Calculates and displays key performance metrics, including accuracy, precision, recall, F1-score, and a confusion matrix.
-   **Command-line Interface:** Easy to run and test from the terminal.

## Technology Stack

-   **Python**
-   **Pandas:** For data manipulation and loading the dataset.
-   **Scikit-learn:** For building the machine learning pipeline (vectorization, model training, and evaluation).

## Setup and Usage

To run this project, you will need Python and pip installed on your system.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/spam-classifier.git](https://github.com/your-username/spam-classifier.git)
    cd spam-classifier
    ```

2.  **Install Dependencies:**
    ```bash
    pip install pandas scikit-learn
    ```

3.  **Download the Dataset:**
    * This model is trained on the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) from the UCI Machine Learning Repository.
    * Download the zip file from the link above and extract it.
    * Find the file named `SMSSpamCollection` and place it in the root directory of your project. Rename it to `spam.csv` for convenience.

4.  **Run the Script:**
    Once the `spam.csv` file is in your project directory, run the classifier from your terminal:
    ```bash
    python spam_classifier.py
    ```

The script will train the model and print the performance evaluation results to the console.
