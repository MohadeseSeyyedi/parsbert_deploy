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
    input_comment = request.form.values()
    prediction = predict(model, input_comment, tokenizer, max_len=128, batch_size=1) # features Must be in the form [[a, b]]

    return render_template('index.html', prediction_text=f'This comment seems to be {prediction}.')


if __name__=='__main__':
    app.run()
