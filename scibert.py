from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
import ipdb
import pickle
from tqdm import trange


def scibert_representation(text):
    # Load the SciBERT model and tokenizer
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Tokenize the input text
    try:
        tokens = tokenizer.tokenize(text)
    except:
        import ipdb
        ipdb.set_trace()

    # Convert tokens to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Add padding if necessary
    max_length = 98
    padding_length = max_length - len(token_ids)
    token_ids += [tokenizer.pad_token_id] * padding_length

    # Convert token IDs to tensor
    input_ids = torch.tensor([token_ids])

    # Pass the input tensor through the model to get word representations
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract the word representations from the output
    word_representations = outputs.last_hidden_state
    return word_representations

df = pd.read_csv('./fragments.csv')

text_dict = dict()
for i in trange(len(df)):
    smiles = df.loc[i, 'frag']
    text = df.loc[i, 'description']
    word_representations = scibert_representation(text)
    # text_dict['smiles'] = torch.mean(word_representations.squeeze(), 0)
    text_dict[smiles] = word_representations.squeeze()

    with open('./block_fragmentation1.pkl', 'wb') as f:
        pickle.dump(text_dict, f)