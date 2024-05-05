import email
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_email_sentiment(email_text):
    # Parse email content
    msg = email.message_from_string(email_text)
    
    # Extract text from email body
    email_body = ""
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            email_body += part.get_payload()

    # Preprocess text: remove special characters, extra whitespaces, etc.
    cleaned_text = re.sub(r'\s+', ' ', email_body)  # Remove extra whitespaces
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove special characters
    
    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(cleaned_text)
    
    # Determine sentiment label based on sentiment score
    if sentiment_score['compound'] >= 0.05:
        sentiment_label = "Positive"
    elif sentiment_score['compound'] <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    
    return sentiment_label, sentiment_score

# Example usage
if __name__ == "__main__":
    example_email = """
    From: sender@example.com
    To: recipient@example.com
    Subject: Example Email

    Hello,

    I hope this email finds you well. Just wanted to check in and see how you're doing.

    Best regards,
    Sender
    """

    sentiment_label, sentiment_score = analyze_email_sentiment(example_email)
    print("Sentiment Label:", sentiment_label)
    print("Sentiment Score:", sentiment_score)
