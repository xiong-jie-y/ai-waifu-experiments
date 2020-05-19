import camphr
import json
import pickle

from annoy import AnnoyIndex

nlp = camphr.load(
    """
lang:
    name: ja_mecab # lang名
pipeline:
    transformers_model:
        trf_name_or_path: bert-base-japanese # モデル名
"""
)

# conversation_examples = [
#     # ("ただいま", "おかえり", ""),
#     ("おやすみなさい。", "おやすみなさい。", "おやすみなさい。.wav"),
#     ("疲れた", "お疲れ様", "otsukaresama.wav"),
#     ("いってきます", "仕事頑張ってね。", "お仕事頑張ってね。.wav")
# ]

conversation_examples = []

for line in open("conversation_pairs.csv", "r").readlines():
    conversation_examples.append(line.split(","))

conversation_data = []

for bob_talk, alice_reply in conversation_examples:
    doc = nlp(bob_talk)
    conversation_data.append(dict(
        embedding=doc.vector.tolist(),
        bob_talk=bob_talk,
        # audio_name=audio_name,
        alice_reply=alice_reply))

json.dump(conversation_data, open("conversations.json", 'w'))

vec_dimension = len(conversation_data[0]['embedding'])
t = AnnoyIndex(vec_dimension, 'angular')  # Length of item vector that will be indexed
for i, conv_exam in enumerate(conversation_data):
    t.add_item(i, conv_exam['embedding'])

print(f"Dimension is {vec_dimension}.")

t.build(10) # 10 trees
t.save('index.ann')