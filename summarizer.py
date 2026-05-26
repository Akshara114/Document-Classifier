from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


def summarize_text(text: str, max_sentences: int = 3) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    # sumy expects a parser with sentences.
    # Use a reasonable limit to avoid extreme compute on very long docs.
    text = text[:50000]
    max_sentences = int(max_sentences) if max_sentences else 3
    if max_sentences < 1:
        max_sentences = 1

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        stemmer = Stemmer("english")
        summarizer = TextRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")
        sentences = summarizer(parser.document, max_sentences)
        return "\n".join(str(s).strip() for s in sentences if str(s).strip()).strip()
    except Exception:
        # Fallback: return a shortened extract (no placeholder).
        return text[:1200].strip()

