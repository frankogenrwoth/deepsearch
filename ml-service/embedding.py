import torch
import torch.nn.functional as F
from model import Model

class Embedding:
    def __init__(self, corpus: list[str]=[]):
        self.embedder = Model.load_embedder()
        self.corpus = corpus

        # self.model = Model.load_model(True)
        # self.tokenizer = Model.load_tokenizer()

    def embed(self):
        corpus_embeddings = self.embedder.encode(self.corpus, convert_to_tensor=True)
        return corpus_embeddings
    

    def normalize(self, vector):
        return F.normalize(vector, p=2, dim=1)


if __name__ == "__main__":
    sentences = ["This is an example sentence", "Each sentence is converted to embedding", "Embedding is normalized", "This is a test", "humor me with this test", "you are a dead man", "I am a a super hero", "dead", "huberman"]
    import time

    start_time = time.time()
    sentences_embedding = Embedding(corpus=sentences).embed()
    end_time = time.time()
    
    print(f"Time taken to load model and tokenizer then embed: {end_time - start_time}")