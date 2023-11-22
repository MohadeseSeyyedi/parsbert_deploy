import torch
from predict import predict
from transformers import BertConfig, BertTokenizer
from architecture import SentimentModel
import numpy as np


config = BertConfig.from_pretrained('models/pretrained/configs', local_files_only=True)
model = SentimentModel(config)
model.load_state_dict(torch.load('models\state_dicts\pytorch_model.bin', map_location=torch.device('cpu')))

# BertModel.from_pretrained('models/pretrained/models', local_files_only=True)
tokenizer = BertTokenizer.from_pretrained('models/pretrained/tokenizers', local_files_only=True)

labels_str = ['SAD', 'HAPPY']
comments = ['غذا بسیار بد بود', 'من لپتاپم را بسیار دوست دارم']
probas = predict(model, comments, tokenizer, max_len=50, batch_size=1)[1]


for i, item in enumerate(probas):
    print(comments[i])
    print(labels_str[np.argmax(item)])