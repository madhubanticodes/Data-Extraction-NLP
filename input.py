import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

# Read the Excel file
df = pd.read_excel('input.xlsx')

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

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Extract article text
    title, article_text = extract_article_text(url)

    if title and article_text:
        # Save the extracted content to a text file
        output_filename = os.path.join(output_folder, f"{url_id}.txt")
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(f"Title: {title}\n\n")
            file.write(article_text)

print("Extraction completed.")