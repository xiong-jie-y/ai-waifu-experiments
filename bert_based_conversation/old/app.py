from flask import Flask, request
import json
from annoy import AnnoyIndex
import camphr

from urllib.parse import urlparse

app = Flask(__name__)

nlp = camphr.load(
    """
lang:
    name: ja_mecab # lang名
pipeline:
    transformers_model:
        trf_name_or_path: bert-base-japanese # モデル名
"""
)

u = AnnoyIndex(768, 'angular')
u.load("index.ann")
conv_pairs = json.load(open("conversations.json"))

@app.route('/kiritan/talk_to')
def get_string_reply():
    message = request.args.get('message')

    closest = u.get_nns_by_vector(nlp(message).vector.tolist(), 1)[0]
    reply = conv_pairs[closest]
    audio_url = 'http://localhost:20020/static/audio/' + reply['audio_name']
    return {
        "reply": reply['alice_reply'],
        "audio_url": audio_url
    }