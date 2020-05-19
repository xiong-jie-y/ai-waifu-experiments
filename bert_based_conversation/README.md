# Bert-based conversational agent UI
This is the streamlit script that enable you to try reply selection algorithm based on similarity between user talk and talk in the database. The reply to the most similar talk from user talk is chosen as a reply.

Currently only bert for japanese is supported.

## How to run it
Firstly they are the modules this script depending on.
I will recommend using conda or pyenv.

```sh
pip install annoy
pip install camphr[mecab]
pip install streamlit
```

You can run this script with the following command.

```sh
wget https://raw.githubusercontent.com/xiong-jie-y/virtual-waifu-experiments/master/bert_based_conversation/example_conversation_pairs.csv

streamlit run https://raw.githubusercontent.com/xiong-jie-y/virtual-waifu-experiments/master/bert_based_conversation/conversation_board.py
```

If it's the first time run, please ingest conversation pair csv file by pressing "Ingest CSV" button. Example file is provided.
