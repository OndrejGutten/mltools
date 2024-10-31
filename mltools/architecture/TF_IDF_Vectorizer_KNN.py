import numpy as np
import pandas as pd
from copy import deepcopy


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import percentileofscore
from random import sample

# TODO: introduce calibrated classifier to the pipeline


class TF_IDF_Vectorizer_KNN():
    def __init__(self, model_name: str, k: int, store_pairwise_distances=False):
        self.__repr__ = f"TF_IDF_Vectorizer_KNN - {model_name}"
        self.k = k
        self.name = model_name
        self.clf = Pipeline((
            ('vect', TfidfVectorizer()),
            ('clf', KNeighborsClassifier(n_neighbors=k, metric='cosine'))
        )
        )
        self.trained = False
        self.store_pairwise_distances = store_pairwise_distances

    def fit(self, X: pd.DataFrame, y: list | pd.Series):
        vectorized_texts = self.clf['vect'].fit_transform(X)
        self.clf['clf'].fit(vectorized_texts, y)
        self.trained = True
        self.known_labels = np.unique(y)

        if self.store_pairwise_distances:
            self.__calculate_pairwise_distances(vectorized_texts, y)

    def __calculate_pairwise_distances(self, vectorized_texts, y, limit=None):
        # Useful for predict_proba
        self.pairwise_distances = {}
        for lbl_a in self.known_labels:
            for lbl_b in self.known_labels:
                key = f'{lbl_a}_{lbl_b}'
                class_a_idxs = np.where(y == lbl_a)[0]
                if limit is not None and len(class_a_idxs) > limit:
                    class_a_idxs = sample(list(class_a_idxs), limit)
                class_b_idxs = np.where(y == lbl_b)[0]
                if limit is not None and len(class_b_idxs) > limit:
                    class_b_idxs = sample(list(class_b_idxs), limit)
                vects_a = vectorized_texts[class_a_idxs]
                vects_b = vectorized_texts[class_b_idxs]
                self.pairwise_distances[key] = cosine_similarity(vects_a, vects_b) if len(
                    class_a_idxs) > 0 and len(class_b_idxs) > 0 else np.array([])

    def predict(self, input_texts: str):
        return self.clf.predict(input_texts)

    def predict_proba(self, input_texts: str):

        if not self.trained:
            raise ValueError("You need to train the model first")

        if not self.store_pairwise_distances:
            return self.clf.predict_proba(input_texts)

        # Custom proba calculation based on similarity to known samples in training data.
        # The largest value is for the class that represents the closest training sample to the examined sample (i.e. KNN with k = 1)
        # The value for any other class X is bigger than 0 only if the examined sample is unusually close to some training sample of class X.
        # More specifically, proba != 0 iff a distance of the examined sample to any training sample of class X is lower than any training sample's distance of the candidate class (KNN1) to any training sample of class X.
        vectorized_texts = self.clf['vect'].transform(input_texts)
        look_around = self.clf['clf'].kneighbors(vectorized_texts)
        max_similarities = 1 - look_around[0][:, 0]
        candidate_lbls = self.clf['clf'].predict(vectorized_texts)
        proba = []
        for candidate_lbl, simscore in zip(candidate_lbls, max_similarities):
            percentile_scores = []
            for lbl in self.known_labels:
                distribution = self.pairwise_distances[f'{candidate_lbl}_{lbl}'].flatten(
                ) if f'{candidate_lbl}_{lbl}' in self.pairwise_distances else np.array([])
                percentilescore = percentileofscore(distribution, simscore)
                score_to_use = 0 if np.isnan(
                    percentilescore) else 100 - round(percentilescore)
                percentile_scores.append(score_to_use)
            if np.sum(percentile_scores) == 0:
                percentile_scores[np.where(
                    candidate_lbl == self.known_labels)[0][0]] = 1
            normalized_scores = np.array(percentile_scores) / \
                np.sum(percentile_scores)
            proba.append(np.array(normalized_scores))
        return np.array(proba)
