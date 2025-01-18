from transformers import AutoTokenizer, AutoModel

MODEL_SAVE_DIR = "ml-service/models/sentence-transformers"

class Model:
  """A model class to lead the model and tokenizer"""

  def __init__(self) -> None:
    pass
  
  def load_model():
    model = AutoModel.from_pretrained(MODEL_SAVE_DIR)
    return model

  def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
    return tokenizer