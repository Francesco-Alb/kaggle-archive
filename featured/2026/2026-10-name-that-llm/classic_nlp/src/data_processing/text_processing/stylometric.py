import re
from collections.abc import Iterable
from lexicalrichness import LexicalRichness
import numpy as np
import pandas as pd


_TOKEN_PATTERN = r"\w+(?:['’]\w+)*|[^\w\s]"
_PUNCTUATION_PATTERNS = {
    "period": re.compile(r"(?<!\.)\.(?!\.)"),
    "ellipsis_3dots": re.compile(r"\.{3}"),
    "ellipsis_char": re.compile("…"),
    "comma": re.compile(","),
    "exclamation": re.compile("!"),
    "question": re.compile(r"\?"),
    "semicolon": re.compile(";"),
    "colon": re.compile(":"),
    "hyphen": re.compile("-"),
    "en_dash": re.compile("–"),
    "em_dash": re.compile("—"),
    "parentheses": re.compile(r"[()]"),
    "brackets": re.compile(r"[\[\]]"),
    "braces": re.compile(r"[{}]"),
    "apostrophe_straight": re.compile("'"),
    "apostrophe_curly": re.compile("’"),
    "quote_straight": re.compile('"'),
    "quote_curly": re.compile("[“”]"),
}

_MARKUP_PATTERNS = {
    "md_bold": re.compile(r"\*\*[^*\n]+\*\*"),
    "md_italic": re.compile(r"(?<!\*)\*(?!\*)[^*\n]+(?<!\*)\*(?!\*)"),
    "md_header": re.compile(r"^#{1,6}\s", re.MULTILINE),
    "md_bullet": re.compile(r"^\s*[-*\u2022]\s+", re.MULTILINE),
    "md_numbered": re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE),
    "md_code_block": re.compile(r"```"),
    "md_inline_code": re.compile(r"(?<!`)`[^`\n]+`(?!`)"),
    "md_table": re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE),
    "md_horizontal_rule": re.compile(r"^\s*(?:---+|\*\*\*+)\s*$", re.MULTILINE),
    "md_link": re.compile(r"\[[^\]]+\]\([^)]+\)"),
    "latex": re.compile(r"\$[^$\n]+\$|\\\(|\\\[|\\frac|\\begin"),
}
_TEXT_PATTERN = {
    "non_ascii": re.compile(r"[^\x00-\x7F]"),
    "curly_quote": re.compile("[‘’“”]"),
    "math_symbol": re.compile("[\u00d7\u00f7\u00b1\u2248\u2260\u2264\u2265\u221e\u2211\u222b\u221a]"),
    "greek_letter": re.compile(r"[\u0370-\u03ff]"),
    "emoji": re.compile(r"[\U0001f300-\U0001faff\u2600-\u27bf]"),
    "nbsp": re.compile("\u00a0"),
    "zero_width": re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]"),
    "double_space": re.compile(r" {2,}"),
    "space_before_punctuation": re.compile(r" [,.;:!?]"),
    "trailing_space": re.compile(r"[ \t]+$", re.MULTILINE),
    "tab": re.compile(r"\t"),
    "newline": re.compile(r"\n"),
    "double_newline": re.compile(r"\n\s*\n"),
}


def _tokenize_words(text: str) -> list[str]:
    """Tokenize words while preserving contractions."""
    tokens = re.findall(_TOKEN_PATTERN, text)

    return [
        token.lower()
        for token in tokens
        if any(char.isalnum() for char in token)
    ]


def _split_sentences(text: str) -> list[str]:
    """Split sentences using terminal punctuation."""
    text = text.strip()

    if not text:
        return []

    return [
        sentence
        for sentence in re.split(
            r"""(?<=[.!?])["'’”)\]]*\s+""",
            text,
        )
        if sentence
    ]


def _lexical_features(words: list[str], mattr_window: int = 15) -> dict[str, float]:
    """Extract lexical diversity and word-length statistics."""
    word_count = len(words)

    if not word_count:
        return {
            "unique_word_count": 0.0,
            "type_token_ratio": 0.0,
            "mattr": np.nan,
            "mtld": np.nan,
            "hdd": np.nan,
            "yules_k": np.nan,
            "average_word_length": 0.0,
            "word_length_std": 0.0,
            "word_length_median": 0.0,
        }

    word_lengths = [len(word) for word in words]
    unique_count = len(set(words))

    # Supply our own tokens to avoid a second tokenization procedure.
    lex = LexicalRichness(
        words,
        preprocessor=None,
        tokenizer=None,
    )

    def safe_metric(method):
        try:
            return float(method())
        except (ValueError, ZeroDivisionError, OverflowError):
            return np.nan

    return {
        "unique_word_count": float(unique_count),
        "type_token_ratio": float(lex.ttr),
        "mattr": safe_metric(
            lambda: lex.mattr(window_size=min(mattr_window, word_count))
        ),
        "mtld": safe_metric(lex.mtld),
        "hdd": safe_metric(lex.hdd),
        "yules_k": safe_metric(lambda: lex.yulek),
        "average_word_length": float(np.mean(word_lengths)),
        "word_length_std": float(np.std(word_lengths)),
        "word_length_median": float(np.median(word_lengths)),
    }


def _sentence_features(
    sentences: list[str],
) -> dict[str, float]:
    """Extract sentence count and sentence-length statistics."""
    if not sentences:
        return {
            "sentence_count": 0.0,
            "average_sentence_length_words": 0.0,
            "sentence_length_std_words": 0.0,
            "sentence_length_median_words": 0.0,
        }

    lengths = [
        len(_tokenize_words(sentence))
        for sentence in sentences
    ]

    return {
        "sentence_count": float(len(sentences)),
        "average_sentence_length_words": float(np.mean(lengths)),
        "sentence_length_std_words": float(np.std(lengths)),
        "sentence_length_median_words": float(np.median(lengths)),
    }


def _punctuation_features(text: str, word_count: int) -> dict[str, float]:
    """Count punctuation choices and normalize them per 100 words."""
    features: dict[str, float] = {}
    for name, pattern in _PUNCTUATION_PATTERNS.items():
        count = len(pattern.findall(text))
        features[f"punct_{name}_count"] = float(count)
        features[f"punct_{name}_per_100_words"] = count * 100 / max(word_count, 1)
    return features


def _formatting_features(text: str) -> dict[str, float]:
    """Measure markdown, LaTeX, unusual characters, and whitespace habits."""
    features: dict[str, float] = {}
    for name, pattern in _MARKUP_PATTERNS.items():
        count = len(pattern.findall(text))
        features[f"format_{name}_count"] = float(count)
        features[f"format_{name}_present"] = float(count > 0)

    for name, pattern in _TEXT_PATTERN.items():
        count = len(pattern.findall(text))
        features[f"text_{name}_count"] = float(count)
        features[f"text_{name}_per_1000_characters"] = count * 1000 / max(len(text), 1)
        features[f"text_{name}_present"] = float(count > 0)
    return features


def extract_text_features(text: str, mattr_window: int = 15) -> dict[str, float]:
    """Extract numeric stylometric features from one passage."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not (
        isinstance(mattr_window, int) or 
        isinstance(mattr_window, bool) or
        mattr_window < 1
        ):
        raise ValueError("mattr_window must be a positive integer")

    if mattr_window < 1:
        raise ValueError("mattr_window must be a positive integer")

    words = _tokenize_words(text)
    sentences = _split_sentences(text)

    features = {
        "word_count": float(len(words)),
        "character_count": float(len(text)),
        "non_whitespace_character_count": float(
            sum(not char.isspace() for char in text)
        ),
        "paragraph_count": float(
            len([
                part for part in re.split(r"\n\s*\n", text.strip())
                if part
            ]) if text.strip() else 0
        ),
        "uppercase_ratio": (
            sum(char.isupper() for char in text if char.isalpha())
            / max(sum(char.isalpha() for char in text), 1)
        ),
        "digit_ratio": (
            sum(char.isdigit() for char in text) / max(len(text), 1)
        ),
    }

    features.update(_lexical_features(words, mattr_window))
    features.update(_sentence_features(sentences))
    features.update(_punctuation_features(text, len(words)))
    features.update(_formatting_features(text))

    return features


def extract_stylometric_features(
    texts: Iterable[str],
    mattr_window: int = 15,
) -> pd.DataFrame:
    """Extract stylometric features for a collection of passages."""
    if isinstance(texts, str):
        raise TypeError("texts must be an iterable of strings, not one string")

    return pd.DataFrame(
        [
            extract_text_features(text, mattr_window)
            for text in texts
        ],
    )