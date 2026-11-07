#Imports & Data Loading

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("spam_ham_dataset.csv")


data.head()

#Exploratory Data Analysis (EDA)

# Count how many times each label (ham, spam) appears in the dataset
class_counts = data['label'].value_counts()
# Plot the class distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution')
plt.ylabel('Frequency')
plt.xlabel('Class Labels')
plt.show()

# Import Counter to easily count word frequencies
from collections import Counter

# Filter and display only the text of the 'ham' emails
data[data['label'] == "ham"]['text']


def get_most_common_words(class_lebel, num_words, colume_name, dataset):
  text = " ".join(message for message in dataset[dataset['label'] == class_lebel][colume_name])
  word = text.lower().split()
  return Counter(word).most_common(num_words)

# Get the top 20 most common words in spam emails and ham emails
spam_words = get_most_common_words('spam', 20, "text", data)
ham_words = get_most_common_words('ham', 20, "text", data)



ham_words


len("this is an spam email")


data['text_length'] = data['text'].apply(len)
data.groupby("label")['text_length'].describe()

# Display the first 5 rows again to see the new 'text_length' column
data.head()

# Import the 'string' module to handle punctuation
import string

# Import a list of common English stop words (e.g., "the", "is", "and")
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# View all punctuation characters
string.punctuation

#text preprocessing
#Define a function to clean the raw text
def preprocess_text(text):
  text = text.lower()
  text = text.translate(str.maketrans("", "", string.punctuation))
  words = text.split()
  words = [word for word in words if word not in ENGLISH_STOP_WORDS]
  text = " ". join(words)
  return text

# View the first raw email text
data['text'].values[0]

# Test the preprocessing function on the first email
preprocess_text(data['text'].values[0])


data['cleaned_text'] = data['text'].apply(preprocess_text)


data.head()

spam_words


data['cleaned_text']

# Import CountVectorizer to convert text into numbers
from sklearn.feature_extraction.text import CountVectorizer


#vectorization
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data['cleaned_text'])
x
# The target variable (0 = Ham, 1 = Spam)
y = data['label_num']
y

# Import train_test_split to split the dataset
from sklearn.model_selection import train_test_split

#Train-Test Split
# Split the data: 80% for training, 20% for testing (random_state ensures reproducibility)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train.shape
x_test.shape

#Import the K-Nearest Neighbors algorithm
from sklearn.neighbors import KNeighborsClassifier

#Training ML Models
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
preds = knn.predict(x_test)

sum(y_test.values == preds)/len(preds)



# Import Naive Bayes, Logistic Regression, and evaluation metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Train and test Naive Bayes
nb = MultinomialNB()
nb.fit(x_train, y_train)
nb_acc = accuracy_score(y_test, nb.predict(x_test))

#Train and test Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(x_test))

# Calculate KNN accuracy using the built-in metric
knn_acc = accuracy_score(y_test, preds)
print(f"KNN: {knn_acc:.4f}")
print(f"Naive Bayes: {nb_acc:.4f}")
print(f"Logistic Regression: {lr_acc:.4f}")
print(classification_report(y_test, nb.predict(x_test)))


#Hugging Face 

from transformers import pipeline

# Load a pre-trained BERT model specifically fine-tuned for spam detection
classifier = pipeline("text-classification",
                      model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

# Define a simple function to detect spam using the Hugging Face model
def detect_spam(message):
    result = classifier(message)[0]
    return result['label']

# Test
print(detect_spam("Free money!!!"))  # spam
print(detect_spam("Meeting tomorrow"))  # ham




#Hugging Face Accuracy Test
test_idx = y_test.index[:200]
sample_texts = data.loc[test_idx, 'text'].tolist()
sample_true = data.loc[test_idx, 'label_num'].values

hf_out = classifier(sample_texts, truncation=True, max_length=128)
hf_preds = [1 if r['label'] == 'LABEL_0' else 0 for r in hf_out]
hf_acc = accuracy_score(sample_true, hf_preds)
print(f"Hugging Face accuracy (200 test emails): {hf_acc:.4f}")
print(f"Majority-class baseline (all ham): {1 - sample_true.mean():.4f}")

# Simple Comparison
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from transformers import pipeline

# Load data
df = pd.read_csv('spam_ham_dataset.csv')

# Prepare data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train scikit-learn
model = MultinomialNB()
model.fit(X, y)

# Test scikit-learn
msg = ["Free money!!!"]
start = time.time()
result_sklearn = model.predict(vectorizer.transform(msg))[0]
time_sklearn = time.time() - start

# Test Hugging Face
classifier = pipeline("text-classification",
                      model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

start = time.time()
result_hf = classifier(msg[0])[0]['label']
time_hf = time.time() - start

# Compare
print("Comparison Results:")
print(f"Scikit-learn: {result_sklearn} (Time: {time_sklearn:.4f}s)")
print(f"Hugging Face: {result_hf} (Time: {time_hf:.4f}s)")



# final accuracy comparison table 
print(pd.DataFrame({
    'Model': ['KNN', 'Naive Bayes', 'Logistic Regression', 'Hugging Face BERT-tiny (200 emails)'],
    'Accuracy': [knn_acc, nb_acc, lr_acc, hf_acc]
}))

