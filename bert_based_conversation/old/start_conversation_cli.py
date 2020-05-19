import logging
import json

from annoy import AnnoyIndex
import camphr

print("Loading conversation data and model.")
nlp = camphr.load(
    """
lang:
    name: ja_mecab
pipeline:
    transformers_model:
        trf_name_or_path: bert-base-japanese
"""
)

u = AnnoyIndex(768, 'angular')
u.load("index.ann")
conv_pairs = json.load(open("conversations.json"))
print("Everything has been loaded.")

try:
    while True:
        print("Please say something.\n>>", end="")
        message = input()

        top_10 = u.get_nns_by_vector(nlp(message).vector.tolist(), 10)
        print(top_10)
        closest = top_10[0]
        reply = conv_pairs[closest]
        print(reply["alice_reply"])

except KeyboardInterrupt:
    print("Finished.")