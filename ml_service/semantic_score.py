from model import Model
from embedding import Embedding
from sentence_transformers import util
import torch

import re

class SemanticScore:
    def __init__(self, data, corpus_embeddings = None):
        self.corpus_embeddings = corpus_embeddings
        self.embedder = Model.load_embedder()
        self.data = data

    def get_semantic_score(self, size, queries: list):
        top_k = min(size, len(queries))

        data = []

        for query in queries:
            query_embedding = Embedding(corpus=query).embed()

            similarity_scores = self.embedder.similarity(query_embedding, self.corpus_embeddings)[0]
            scores, indices = self.get_top_k_indices(similarity_scores, k=top_k)

            data.append((query, scores, indices))

        return data


    def get_top_k_indices(self, similarity_scores, k):
        scores, indices = torch.topk(similarity_scores, k=k)

        return scores, indices

    def display_results(self, result):
        print("query: ", result[0])
        print()
        print("RESULTS:\n\n")

        for score, idx in zip(result[1], result[2]):
            print(self.data[idx], f"(Score: {score:.4f})")
