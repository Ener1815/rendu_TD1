
import ast

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import numpy as np
import pickle
import string
nltk.download('punkt')
# Téléchargement des stopwords
nltk.download('stopwords')
stop_words = list(stopwords.words('french'))
stemmer = FrenchStemmer()




def tokenize_and_stem(text):

    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    for token in tokens:
        if token.isalpha():
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def preprocess_part2(df):

    features_list = []
    labels_list = []
    tokens_list = []

    for index, row in df.iterrows():
        # Get the tokens from the "tokens" column
        tokens = ast.literal_eval(row['tokens'])
        tokens_list.append(tokens)
        individual_labels = ast.literal_eval(row['is_name'])

        for i, token in enumerate(tokens):
            previous_word = tokens[i - 1] if i > 0 else None
            next_word = tokens[i + 1] if i < len(tokens) - 1 else None

            # Extract features for the token
            features = {
                "is_capitalized": token[0].isupper() if token else False,
                "is_previous_capitalized": previous_word[0].isupper() if previous_word and previous_word != "" else False,
                "is_next_capitalized": next_word[0].isupper() if next_word and next_word != "" else False,
                "length": len(token),
                "position": i,
            }

            # Append the features and the corresponding label to the lists
            features_list.append(features)
            labels_list.append(individual_labels[i])

    return features_list, labels_list, tokens_list

def make_features(df, task, train_mode=True, vectorizer_path=None):

    if task == "is_comic_video":
        if train_mode:
            y = get_output(df, task)
        else:
            y = None  # If it's not training mode, we don't have the target column

        if train_mode or vectorizer_path is None:
            tfidf_vectorizer = TfidfVectorizer(
                max_features=200000,
                stop_words=stop_words,
                use_idf=True,
                tokenizer=tokenize_and_stem,
                ngram_range=(1, 3)
            )
            X = tfidf_vectorizer.fit_transform(df['video_name'])
            # Save the vectorizer
            with open(vectorizer_path if vectorizer_path else 'tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(tfidf_vectorizer, f)
        else:
            # Load the vectorizer for prediction
            with open(vectorizer_path, 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            X = tfidf_vectorizer.transform(df['video_name'])

        if train_mode:
            return X, y
        else:
            return X,

    else :
        features_list, labels_list, tokens_list = preprocess_part2(df)
        return features_list, labels_list, tokens_list

def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y
