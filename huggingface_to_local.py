from transformers import BertModel, BertConfig, BertTokenizer

def load_model(huggingface_path, local_path):
    BertModel.from_pretrained(huggingface_path).save_pretrained(local_path)

def load_config(huggingface_path, local_path):
    BertConfig.from_pretrained(huggingface_path).save_pretrained(local_path)

def load_tokenizer(huggingface_path, local_path):
    BertTokenizer.from_pretrained(huggingface_path).save_pretrained(local_path)

