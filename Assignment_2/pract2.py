import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data
documents = [
    "I went to river bank",

    "I am going to bank to deposit money",
]

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Convert to array for visualization
print(tfidf_matrix.toarray())

# Display feature names
print(tfidf_vectorizer.get_feature_names_out())


nltk.download('punkt')

# Tokenize sentences into words
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

# Get word vectors for a specific word
print(model.wv['bank'])

# Get most similar words
print(model.wv.most_similar('bank'))


