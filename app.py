import torch
from transformers import BertConfig, BertTokenizer

from flask import Flask, request, render_template
from predict import predict
from architecture import SentimentModel


#Create an app object using the Flask class. 
app = Flask(__name__)

model_path = 'models/state_dicts/pytorch_model.bin'
config_path = 'models/pretrained/configs'
tokenizer_path = 'models/pretrained/tokenizers' 

config = BertConfig.from_pretrained(config_path, local_files_only=True)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

model = SentimentModel(config)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def show_prediction():
    labels_str = ['SAD', 'HAPPY']
    input_comment = [x for x in request.form.values()][0]
    prediction = predict(model, [input_comment], tokenizer, max_len=128, batch_size=1)[1]

    from numpy import argmax
    prediction_text = f'This comment seems to be {labels_str[argmax(prediction)]}.'
    return render_template('index.html', prediction_text=prediction_text, input_comment=input_comment)


if __name__=='__main__':
    app.run()
