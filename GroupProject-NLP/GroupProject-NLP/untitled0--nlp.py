import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import re

# 1. Load Dataset
df = pd.read_csv('test 2(in).csv')

# 2. Data Cleaning Function
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove numbers/punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning
df['Cleaned_Story'] = df['Story'].apply(clean_text)

# 3. Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive', analysis.sentiment.polarity
    elif analysis.sentiment.polarity < 0:
        return 'Negative', analysis.sentiment.polarity
    else:
        return 'Neutral', analysis.sentiment.polarity

# Apply function
df[['Sentiment', 'Sentiment_Score']] = df['Cleaned_Story'].apply(
    lambda x: pd.Series(analyze_sentiment(x))
# 4. Results & Visualization
sentiment_counts = df['Sentiment'].value_counts()
print(sentiment_counts)

# Save results
df.to_csv('sentiment_analysis_results.csv', index=False)

# Plot
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution')
plt.show()


