import re
import string


def simple_text_cleaner(text: str) -> str:
    """Lowercase text and remove common markup, URLs, punctuation, and digits."""
    text = text.lower()                                                 # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)                                 # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)                   # Remove URLs
    text = re.sub(r'<.*?>+', '', text)                                  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)    # Remove punctuation
    text = re.sub(r'\n', '', text)                                      # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)                                # Remove words containing numbers
    return text
