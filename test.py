import torch
from predict import predict
from transformers import BertConfig, BertTokenizer
from architecture import SentimentModel


config = BertConfig.from_pretrained('models/pretrained/configs', local_files_only=True)
model = SentimentModel(config)
model.load_state_dict(torch.load('models\state_dicts\pytorch_model.bin', map_location=torch.device('cpu')))

# BertModel.from_pretrained('models/pretrained/models', local_files_only=True)
tokenizer = BertTokenizer.from_pretrained('models/pretrained/tokenizers', local_files_only=True)


comments = ['غذا بسیار بد بود', 'من لپتاپم را بسیار دوست دارم']
output = predict(model, comments, tokenizer, max_len=50, batch_size=1)

print(output)