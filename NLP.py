# ------------------------------------------------------------
# 0. Setup
# ------------------------------------------------------------
import nltk, spacy, torch, numpy as np, pandas as pd
from datasets import load_dataset
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')

# tiny sentiment corpus (2000 rows) ---------------------------------
ds = load_dataset("imdb", split="train[:2000]+test[:200]")
df = ds.to_pandas()[["text", "label"]].sample(frac=1, random_state=42)
train_df, test_df = df.iloc[:1600], df.iloc[1600:]

print("📦 Data shape", train_df.shape, "sample label:", train_df["label"].iloc[0])

# ------------------------------------------------------------
# 1. Tokenisation (white-space & NLTK word) --------------------------
sample = train_df["text"].iloc[0][:200]
print("\n🔡 1. Tokenisation")
print("White-space:", sample.split()[:10])
print("NLTK word  :", nltk.word_tokenize(sample)[:10])

# ------------------------------------------------------------
# 2. Lower-casing & punctuation removal -----------------------------
import string, re
def clean(txt):
    txt = txt.lower()
    txt = re.sub(f"[{string.punctuation}]", " ", txt)
    return txt
train_df["clean"] = train_df["text"].apply(clean)
print("\n🧹 2. Clean text:", clean(sample)[:120])

# ------------------------------------------------------------
# 3. Stop-word removal ----------------------------------------------
from nltk.corpus import stopwords
stop = set(stopwords.words("english"))
train_df["no_stop"] = train_df["clean"].apply(
    lambda x: " ".join([w for w in x.split() if w not in stop]))
print("\n🛑 3. No stop   :", train_df["no_stop"].iloc[0][:120])

# ------------------------------------------------------------
# 4. Stemming (Porter) & Lemmatisation (spaCy) ----------------------
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
train_df["stem"] = train_df["no_stop"].apply(
    lambda x: " ".join([stemmer.stem(w) for w in x.split()]))
print("\n🌱 4. Stemmed   :", train_df["stem"].iloc[0][:120])

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
train_df["lemma"] = train_df["no_stop"].apply(
    lambda x: " ".join([tok.lemma_ for tok in nlp(x)]))
print("Lemmatised :", train_df["lemma"].iloc[0][:120])

# ------------------------------------------------------------
# 5. Bag-of-Words (Count) ------------------------------------------
vectorizer = CountVectorizer(max_features=1000)
X_train_bow = vectorizer.fit_transform(train_df["lemma"])
X_test_bow  = vectorizer.transform(test_df["lemma"].apply(clean).apply(
    lambda x: " ".join([tok.lemma_ for tok in nlp(x)])))
clf_bow = LogisticRegression(max_iter=1000)
clf_bow.fit(X_train_bow, train_df["label"])
print("\n📊 5. BoW accuracy:", accuracy_score(test_df["label"], clf_bow.predict(X_test_bow)))

# ------------------------------------------------------------
# 6. TF-IDF ---------------------------------------------------------
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(train_df["lemma"])
X_test_tfidf  = tfidf.transform(test_df["lemma"].apply(clean).apply(
    lambda x: " ".join([tok.lemma_ for tok in nlp(x)])))
clf_tfidf = LogisticRegression(max_iter=1000)
clf_tfidf.fit(X_train_tfidf, train_df["label"])
print("TF-IDF accuracy :", accuracy_score(test_df["label"], clf_tfidf.predict(X_test_tfidf)))

# ------------------------------------------------------------
# 7. Word2Vec (gensim) – tiny demo ----------------------------------
from gensim.models import Word2Vec
sentences = [s.split() for s in train_df["lemma"]]
w2v = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4, epochs=10)
print("\n🌍 7. Word2Vec – similarity (good ↔ bad):",
      w2v.wv.similarity("good", "bad"))

# average word2vec feature matrix
def sent_vec(s):
    words = [w for w in s.split() if w in w2v.wv]
    return np.mean([w2v.wv[w] for w in words], axis=0) if words else np.zeros(50)

X_train_w2v = np.vstack(train_df["lemma"].apply(sent_vec))
X_test_w2v  = np.vstack(test_df["lemma"].apply(clean).apply(
    lambda x: " ".join([tok.lemma_ for tok in nlp(x)])).apply(sent_vec))
clf_w2v = LogisticRegression(max_iter=1000)
clf_w2v.fit(X_train_w2v, train_df["label"])
print("Word2Vec accuracy:", accuracy_score(test_df["label"], clf_w2v.predict(X_test_w2v)))

# ------------------------------------------------------------
# 8. Transformer (zero-shot) ---------------------------------------
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
preds = classifier(test_df["text"].iloc[:20].tolist())
hits = sum([p['label'] == 'POSITIVE' for p in preds]) == (test_df["label"].iloc[:20].sum())
print("\n🤖 8. Transformer (20 samples) POS accuracy ≈", hits/20)

# ------------------------------------------------------------
# 9. Token counts & vocab size -------------------------------------
tokens = [tok for s in train_df["lemma"] for tok in s.split()]
cnt = Counter(tokens)
print("\n📏 9. Vocab size:", len(cnt), "  Top-5 tokens:", cnt.most_common(5))

# ------------------------------------------------------------
# 10. One-liner cheat reminder --------------------------------------
print("""
🧠 NLP Pipeline Cheat
1.  Raw text
2.  Clean (lower, punct)
3.  Tokenise
4.  Remove stop-words
5.  Stem / Lemma
6.  Vectorise (Bow, TF-IDF, Word2Vec, Transformer)
7.  Model (LogReg, BERT, etc.)
8.  Evaluate
""")



