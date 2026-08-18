import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.utils import logging as hf_logging
from sklearn.metrics.pairwise import cosine_similarity


# ── Constants ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
SENTIMENT_MODEL    = "cardiffnlp/twitter-roberta-base-sentiment-latest"
KNOWLEDGE_BASE_FILE = "knowledge_base.csv"
ESCALATION_THRESHOLD = 0.90


# ── 1. Knowledge Base Loading ─────────────────────────────────────────────────
def loadKnowledgeBase(file):
    """Load questions and answers from a CSV file using pandas."""
    try:
        dataFrame = pd.read_csv(file)
        questions = dataFrame['question'].tolist()
        answers   = dataFrame['answer'].tolist()
        return questions, answers
    except FileNotFoundError:
        print("File path not found!")
        exit()
    except KeyError:
        print("CSV must have 'question' and 'answer' columns!")
        exit()
    except pd.errors.ParserError as e:
        print("CSV format error. Ensure entries with commas are enclosed in quotes.")
        print(f"Details: {e}")
        exit()


# ── 2. Model Loading ──────────────────────────────────────────────────────────
def loadModels():
    """Load the embedding model and sentiment analysis pipeline."""
    try:
        hf_logging.set_verbosity_error()
        model     = loadEmbeddingModel()
        sentiment = loadSentimentPipeline()
        return model, sentiment
    except Exception as e:
        print("ERROR: Could not load models. Check dependencies.")
        print(f"Details: {e}")
        exit()


def loadEmbeddingModel():
    """Load SentenceTransformer; try local cache first, then download."""
    try:
        return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except Exception:
        return SentenceTransformer(EMBEDDING_MODEL)


def loadSentimentPipeline():
    """
    Load a 3-class (negative/neutral/positive) sentiment model.
    Uses cardiffnlp/twitter-roberta-base-sentiment-latest.
    Tries local cache first, then downloads.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL, local_files_only=True)
        model     = AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL, local_files_only=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        model     = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# ── 3. Embedding Generation ───────────────────────────────────────────────────
def generateEmbeddings(questions, model):
    """Convert all knowledge-base questions into vector embeddings."""
    return model.encode(questions)


# ── 4. Semantic Search ────────────────────────────────────────────────────────
def findBestMatch(userInput, storedEmbeddings, model):
    """
    Encode the user query and find the most similar knowledge-base question
    using cosine similarity.
    Returns the index of the best match.
    """
    queryEmbedding = model.encode(userInput).reshape(1, -1)
    similarities   = cosine_similarity(queryEmbedding, storedEmbeddings)
    bestIndex      = np.argmax(similarities[0])
    return int(bestIndex)


# ── 5. Sentiment Analysis ─────────────────────────────────────────────────────
def analyzeSentiment(userInput, sentiment):
    """
    Run sentiment analysis on the user input.
    Returns a normalized label (NEGATIVE / NEUTRAL / POSITIVE) and confidence score.
    """
    result = sentiment(userInput)
    label  = normalizeSentimentLabel(result[0]['label'])
    score  = result[0]['score']
    return label, score


def normalizeSentimentLabel(label):
    """
    Map raw model output labels to human-readable strings.
    cardiffnlp model returns label_0 / label_1 / label_2.
    """
    labelMap = {
        "label_0": "NEGATIVE",
        "label_1": "NEUTRAL",
        "label_2": "POSITIVE",
        "negative": "NEGATIVE",
        "neutral":  "NEUTRAL",
        "positive": "POSITIVE",
    }
    return labelMap.get(label.lower(), label.upper())


# ── 6. Escalation Logic ───────────────────────────────────────────────────────
def shouldEscalate(label, score):
    """Escalate if sentiment is strongly negative (confidence > threshold)."""
    return label == "NEGATIVE" and score > ESCALATION_THRESHOLD


# ── 7. Conversation Loop ──────────────────────────────────────────────────────
def runChatLoop(questions, answers, storedEmbeddings, model, sentiment):
    """
    Main interactive loop.
    Accepts user input, performs sentiment analysis, retrieves the best answer,
    and maintains a session history printed as a summary on exit.
    """
    history = []

    while True:
        question = input("You: ").strip()

        if question.lower() == "quit":
            printSessionSummary(history)
            print("Bye-Bye")
            break

        if question == "":
            print("Please type a question.")
            continue

        # Sentiment analysis
        label, score = analyzeSentiment(question, sentiment)
        print(f"Sentiment: {label} ({score:.2f})")

        # Escalation check
        escalated = shouldEscalate(label, score)
        if escalated:
            print("We recommend contacting a human advisor")

        # Semantic search → answer
        i      = findBestMatch(question, storedEmbeddings, model)
        answer = answers[i]
        print(f"Answer: {answer}\n")

        # Record in history
        history.append({
            "question":  question,
            "sentiment": label,
            "score":     score,
            "answer":    answer,
            "escalated": escalated,
        })


def printSessionSummary(history):
    """Print a summary of all questions asked during the session."""
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)

    if not history:
        print("No questions were asked this session.")
        return

    for number, item in enumerate(history, start=1):
        escalation = "Yes" if item["escalated"] else "No"
        print(f"\n{number}. Question : {item['question']}")
        print(f"   Sentiment: {item['sentiment']} ({item['score']:.2f})")
        print(f"   Escalated: {escalation}")
        print(f"   Answer   : {item['answer']}")

    print("="*60 + "\n")


# ── OOP Basics ────────────────────────────────────────────────────────────────
class Assistant:
    def __init__(self, file):
        self.questions, self.answers = loadKnowledgeBase(file)
        self.model, self.sentiment = loadModels()
        self.storedEmbeddings = generateEmbeddings(self.questions, self.model)

    def run(self):
        runChatLoop(self.questions, self.answers, self.storedEmbeddings, self.model, self.sentiment)


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("     Welcome to Student Support AI")
    print("=" * 60)
    print("Type 'quit' to exit.\n")

    assistantBot = Assistant(KNOWLEDGE_BASE_FILE)
    assistantBot.run()


if __name__ == "__main__":
    main()
