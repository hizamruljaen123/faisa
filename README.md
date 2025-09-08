# Sentiment Analysis of Tokopedia Product Reviews Using SVM

This project demonstrates how to classify Tokopedia product reviews into **positive**, **negative**, or **neutral** sentiments using **Support Vector Machine (SVM)**.

---

## Steps Overview

1. **Data Collection**  
   - Scrape or collect Tokopedia product reviews.  
   - Each review should have a **text** and **rating** (optional for labeling).

2. **Data Preprocessing**  
   - Lowercase text, remove punctuation and stopwords.  
   - Tokenize and vectorize text using **TF-IDF**.  
   - Encode labels: Positive (1), Negative (0), Neutral (2).

3. **SVM Model**  
   - Split dataset into **training** and **testing** sets.  
   - Train **SVM classifier** with linear kernel.  
   - Evaluate accuracy, precision, recall, and F1-score.

4. **Prediction & Evaluation**  
   - Predict sentiment for new reviews.  
   - Analyze results to identify common positive/negative feedback.

---

## Python Implementation (Simplified)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load Tokopedia reviews dataset
data = pd.read_csv('tokopedia_reviews.csv')  # columns: 'review', 'sentiment'

# Features and labels
X = data['review']
y = data['sentiment']  # Positive, Negative, Neutral

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# SVM Classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
