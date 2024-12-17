import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import html
from bs4 import BeautifulSoup
import re
from nltk.stem import WordNetLemmatizer
import os

# Downloading NLTK data
nltk.download('stopwords') # to remove common words like this, and, that ...
nltk.download('punkt') # Breaks the tweets into smaller portions (sentences or words)
nltk.download('wordnet') # Dictionary that helps finding word meanings or synonims 


# Use panda to load the dataset containing the tweets and labels
try:
    df = pd.read_csv('labeled_data.csv', encoding='utf-8')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: labeled_data.csv not found. Ensure the file exists in the content-moderator-main directory.")
    exit()
except pd.errors.ParserError as e:
    print(f"Error loading the dataset: {e}")
    exit()

# Return only the necessary columns 
df = df[['class', 'tweet']]

# Preprocess text
def preprocess_text(text):
    '''This function processes text by decoding HTML entities, converting to lowercase, removing HTML tags, punctuation, emojis, and numbers. 
        It tokenizes the text, removes stopwords and short words, and lemmatizes the words to their base form.

        -- Input: A string of text.
        -- Output: A preprocessed string with cleaned and normalized text.'''

    # check if the input is in string form
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)  # Decode HTML entities
    text = text.lower()  # Convert to lowercase
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
     # Remove emojis
    emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # Emoticons
                                u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                                u"\U00002500-\U00002BEF"  # Chinese characters
                                u"\U00002702-\U000027B0"  # Dingbats
                                u"\U000024C2-\U0001F251"  # Enclosed characters
                                "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)  # Remove emojis
    words = word_tokenize(text)  # Tokenize text into individual words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 1]  # Remove stopwords and short words

    # Lemmatization: to convert words to its base form, e.g. running = run
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
    
    return ' '.join(words)

# Apply preprocessing to the 'tweet' column
print("Starting preprocessing...")
df['tweet'] = df['tweet'].apply(preprocess_text)
print("Preprocessing completed.")

# Save processed dataset
output_path = 'preprocessed_hate_speech_dataset.csv'
# Save the processed DataFrame to the specified path
df.to_csv(output_path, index=False)
print(f"Preprocessed dataset saved at {output_path}.")