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

# example usage
# semantic_score = SemanticScore()
# cos_scores = semantic_score.get_semantic_score(["Frank", "what is the book about"])
# indices = semantic_score.get_top_k_indices(cos_scores, 5)
# semantic_score.display_results(["Frank", "what is the book about"], passages, indices)

if __name__ == "__main__":
    # sample embeddings, sample passages about music featuring micheal jackson
    sentences = [
        "The teachers and staff at the special school, Graysmill, Did what they could to give the severes a life afterwards, And they presumed I would be accepted to work, At the CALL Centre of Edinburgh University, for a long time to lurk.",
        "It’s now CALL Scotland, and researches special tech, Develops assistive software, devices, and communication aids;It digitalise written exams energetically and with voice, For disabled kids who need to have their own writing choice.",
        "But I went to Daniel Stewarts nursery, was well accepted, superior, As I came top of the class for both words and numbers, And as it is a top private school near Edinburgh’s city centre, I found the sympathy hard at Graysmill ‘cos I was not inferior.", "In the 70s and 80s they thought the special pupils couldn’t interact, In mainstream schools where the able-bodied were understood; Most of my friends had a dislike of normal, ordinary kids, And didn’t understand my perceptions of relationality and brotherhood.",
        "So as it was sometimes an effort for me to be part of the school, And I just wanted to walk away from all things disabled or impaired,The moment I started university where opportunity beckoned, Where my intentions and abilities could be so aired. ",
        "I wanted to maybe be a software engineer for organisations, But knew I couldn’t type all day every day with my foot, So after uni got a part-time job at the CALL Centre, but felt self-defeated, ‘Cos I'd had blows with my parents about my own mechanism of input.",
        "I did home computing growing up using my hands on the keyboard, But did my school and homework with my foot, not good, And since they wanted me to go to university, no big deal, They forced me to keep using the faster mechanism, the switch for my foot.",
        "So I resented the CALL Centre right throughout my young years, For not believing or ingratiating me when I told them of my hand dexterity,And as a graduate able to deliberate upon my case of disrespect, I can say that my parents should have certainly been certified for neglect. ",
        "I did not renew my contract with the Call, was only for four months, As I didn’t want to put myself through that close contact and innocence assumption, But think that they do an note-worthy job for severely disabled kids, And that my case was an exception to their loving, kind gumption.",
    ]

    # generate sample embeddings
    embeddings = Embedding(corpus=sentences).embed()

    print("embeddings shape: ", embeddings.shape)

    # sample queries
    queries = ["what does a software engineer do", "what is a contract"]

    # instantiate the semantic score class
    semantic_score = SemanticScore(data=sentences, corpus_embeddings=embeddings)

    outputs = semantic_score.get_semantic_score(size=5, queries=queries)

    for i in outputs:
        semantic_score.display_results(i)
        print()
