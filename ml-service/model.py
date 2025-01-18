from transformers import AutoTokenizer, AutoModel

from sentence_transformers import SentenceTransformer

MODEL_SAVE_DIR = "ml-service\\models\\sentence-transformers"

class Model:
  """A model class to lead the model and tokenizer"""

  def __init__(self) -> None:
    pass
  
  def load_model(with_sentence_transformer=False):
    if with_sentence_transformer:
      model = SentenceTransformer(MODEL_SAVE_DIR)
      
    else:
      model = AutoModel.from_pretrained(MODEL_SAVE_DIR)
      
    return model
  

  def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
    return tokenizer