import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import matplotlib.pyplot as plt

# Load preprocessed dataset with pandas
df = pd.read_csv('preprocessed_hate_speech_dataset.csv')

# Assign columns to X (data) and Y (labels)
X = df['tweet']
y = df['class']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# This will vectorize the text into numerical data
vectorizer = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.8, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)

# Create and train the model (Multinomial Naive Bayes classifier)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Hate Speech", "Offensive Speech", "Accepted Speech"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Save the model and vectorizer as .pkl files
with open('content_moderator_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)


# Load the trained model and vectorizer
with open('content_moderator_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# The labels of the tweets indicate the next 0 - hate, 1 - offensive, 2 - accepted speech. 
# We will map them into our text outputs
classification_map = {0: "Hate Speech", 1: "Offensive Speech", 2: "Accepted Speech"}

print("Type 'quit' to exit.")

while True:
    # Take input from the user
    comment = input("\nEnter a comment to classify (or type 'quit' to exit): ").strip()

    # Check for exit condition
    if comment.lower() == 'quit':
        print("Goodbye!")
        break

    # Check if input is valid
    if not comment:
        print("Error: No text provided. Please enter a valid comment.")
        continue

    # Preprocess and vectorize the input text
    text_tfidf = vectorizer.transform([comment])
    prediction = model.predict(text_tfidf)[0]

    # Output the classification
    print(f"Classification: {classification_map.get(prediction, 'Unknown')}")
