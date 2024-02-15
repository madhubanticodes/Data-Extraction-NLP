import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load stopwords for English
stop_words = set(stopwords.words("english"))

# Create a folder to store text files
output_folder = 'extracted_articles'
os.makedirs(output_folder, exist_ok=True)

# Function to extract article text from a given URL
def extract_article_text(url):
    try:
        response = requests.get(url)
        html_content = response.text

        # Use BeautifulSoup to parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract title and text (modify this based on HTML structure)
        title = soup.title.text.strip()
        article_text = ' '.join([p.text for p in soup.find_all('p')])

        return title, article_text
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None, None

# Function to perform textual analysis and compute variables
def analyze_text(article_text):
    # Tokenize the text
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(article_text)

    # Remove stopwords
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

    # Compute word frequency
    word_freq = FreqDist(filtered_tokens)

    # Compute sentiment scores
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(article_text)
    positive_score = sentiment_score['pos']
    negative_score = sentiment_score['neg']
    polarity_score = sentiment_score['compound']

    # Compute subjectivity score
    subjectivity_score = TextBlob(article_text).sentiment.subjectivity

    # Tokenize sentences
    sentences = sent_tokenize(article_text)

    # Compute average sentence length
    avg_sentence_length = len(tokens) / len(sentences)

    # Compute percentage of complex words
    complex_words = [word for word in tokens if syllable_count(word) >= 3]
    percentage_complex_words = (len(complex_words) / len(tokens)) * 100

    # Compute Fog Index
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    # Compute average number of words per sentence
    avg_words_per_sentence = len(tokens) / len(sentences)

    # Compute complex word count
    complex_word_count = len(complex_words)

    # Compute syllables per word
    syllables_per_word = sum(syllable_count(word) for word in tokens) / len(tokens)

    # Compute personal pronouns
    personal_pronouns = sum(1 for word in tokens if word.lower() in ['i', 'me', 'my', 'mine', 'myself'])

    # Compute average word length
    avg_word_length = sum(len(word) for word in tokens) / len(tokens)

    return positive_score, negative_score, polarity_score, subjectivity_score, \
           avg_sentence_length, percentage_complex_words, fog_index, \
           avg_words_per_sentence, complex_word_count, len(tokens), \
           syllables_per_word, personal_pronouns, avg_word_length

# Function to count syllables in a word
def syllable_count(word):
    count = 0
    vowels = "aeiouy"
    word = word.lower().strip(".:;?!")
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

# Load the Excel file with URLs and IDs
df = pd.read_excel('input.xlsx')

# Load the output structure Excel file
output_structure_df = pd.read_excel('Output Data Structure.xlsx')

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Extract article text
    title, article_text = extract_article_text(url)

    if title and article_text:
        # Analyze the text and compute variables
        variables = analyze_text(article_text)

        # Create a dictionary with computed variables
        output_data = {
            'URL_ID': url_id,
            'POSITIVE SCORE': variables[0],
            'NEGATIVE SCORE': variables[1],
            'POLARITY SCORE': variables[2],
            'SUBJECTIVITY SCORE': variables[3],
            'AVG SENTENCE LENGTH': variables[4],
            'PERCENTAGE OF COMPLEX WORDS': variables[5],
            'FOG INDEX': variables[6],
            'AVG NUMBER OF WORDS PER SENTENCE': variables[7],
            'COMPLEX WORD COUNT': variables[8],
            'WORD COUNT': variables[9],
            'SYLLABLE PER WORD': variables[10],
            'PERSONAL PRONOUNS': variables[11],
            'AVG WORD LENGTH': variables[12]
        }

        # Create a DataFrame from the dictionary
        output_df = pd.DataFrame([output_data])

        # Append the computed variables to the output structure DataFrame
        output_structure_df = pd.concat([output_structure_df, output_df], ignore_index=True)

# Save the output structure DataFrame to a new Excel file
output_structure_df.to_excel('computed_variables_output.xlsx', index=False)

print("Textual analysis and variable computation completed.")
