import torch
import torch.nn.functional as F
from model import Model


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Embedding:
    def __init__(self):
        self.model = Model.load_model(True)
        self.tokenizer = Model.load_tokenizer()

    def get_embedding(self, sentences):
        # encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

        encoded_input = self.model.encode(sentences, convert_to_tensor=True)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        return mean_pooling(model_output, encoded_input['attention_mask'])
    

    def normalize(self, vector):
        return F.normalize(vector, p=2, dim=1)
    
    def embed(self, sentences):
        embedding = self.get_embedding(sentences)
        normalized_embedding = self.normalize(embedding)

        return normalized_embedding


if __name__ == "__main__":
    sentences = ["This is an example sentence", "Each sentence is converted to embedding", "Embedding is normalized", "This is a test", "humor me with this test", "you are a dead man", "I am a a super hero", "dead", "huberman"]
    import time

    start_time = time.time()
    embedding = Embedding()
    end_time = time.time()
    sentences_embedding = embedding.embed(sentences)
    print(f"Time taken to load model and tokenizer then embed: {end_time - start_time}")