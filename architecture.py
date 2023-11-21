
import torch.nn as nn
from transformers import BertModel

pretrained_model_path = 'models/pretrained/models'

class SentimentModel(nn.Module):
    def __init__(self, config):
        super(SentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path, local_files_only=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits