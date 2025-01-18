from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from sentence_transformers import SentenceTransformer

MODEL_SAVE_DIR = "ml-service/models/sentence-transformers"

print(MODEL_SAVE_DIR)

class Model:
  """A model class to lead the model and tokenizer"""

  def __init__(self) -> None:
    pass

  def load_embedder():
    embedder = SentenceTransformer(MODEL_SAVE_DIR)

    return embedder
  
# if __name__ == "__main__":
#   embedder = Model.load_embedder()
#   print(embedder)