import ipdb
import re
import signal
import openai

import pandas as pd
from contextlib import contextmanager
from openai import OpenAI
from tqdm import tqdm, trange
from datetime import datetime

client = OpenAI(api_key = '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')

def api_call(smiles):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"You are a chemist. Provide a brief description of the physical and chemical properties of this molecular fragment: {smiles}. Keep it under 100 words."},
            # {"role": "user", "content": f"Pretend that you are a chemist working on identifying toxic molecules. Can you provide information about the structure and metabolism of the following fragment of a molecule and its contribution to the toxicity of the compound containing it? Try to be brief and specific; do not provide neutral information. Here is the SMILES string of the fragment: {smiles}. This is an example: Structure: The presence of a carboxylate group indicates an acidic moiety, which can influence biological interactions. Metabolism: Conjugation reactions may occur to facilitate excretion but can also lead to toxic metabolites if overwhelmed. Contribution to Toxicity: The acidic nature of the carboxylate group may disrupt cellular processes, contributing to toxicity."},
            # {"role": "user", "content": "For example,"},
        ]
    )
    return response.choices[0].message.content

df = pd.read_csv('./fragments.csv')
# SMILES, DESCRIPTION = [], []
for i in trange(len(df)):
    smiles = df.loc[i, 'frag']
    text = api_call(smiles)
    print(f'SMILES: {smiles} \n {text}')
    df.loc[i, 'description'] = text
    df.to_csv('./fragments.csv', index=False)


