from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")

english_bot.set_trainer(ChatterBotCorpusTrainer)
# english_bot.train("chatterbot.corpus.english")


@app.route("/talk", methods=['POST'])
def get_bot_response():
    req_data = request.get_json()
    print(req_data)
    message = req_data['message']
#    userText = request.args.get('msg')
    return str(english_bot.get_response(message))

@app.route("/talk/finance", methods=['POST'])
def get_finance_bot_response():
    req_data = request.get_json()
    print(req_data)
    message = req_data['message']
#    userText = request.args.get('msg')
    return str(english_bot.get_response(message))

@app.route("/talk/insurance", methods=['POST'])
def get_insurance_bot_response():
    req_data = request.get_json()
    print(req_data)
    message = req_data['message']
#    userText = request.args.get('msg')
    return str(english_bot.get_response(message))


if __name__ == "__main__":
    app.run()

