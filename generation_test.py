# Standard Libraries
import random
import math
import time
from collections import Counter, defaultdict

# Third-Party Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.legacy.data import Field, BucketIterator
from torchtext.vocab import vocab

# RDKit for Molecular Handling
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs

# Model-Specific Imports
from model import vocabulary, seq2seq_attention, multi_gen_test
from model.seq2seq_dataset import SMILESDataset

# Utilities
import pickle
import tqdm

# Load the model and dataset
SRC = Field(tokenize=vocabulary.SMILESTokenizer().tokenize, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=False, 
            batch_first=True)
TRG = Field(tokenize=vocabulary.SMILESTokenizer().tokenize, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=False, 
            batch_first=True)

# Load vocabulary
src_vocab_path = '/home/wei/Desktop/Similarity/chembl33_dataset/vocab_pkls/stereo_experiment_vocab.pkl'
trg_vocab_path = '/home/wei/Desktop/Similarity/chembl33_dataset/vocab_pkls/stereo_experiment_vocab.pkl'
with open(src_vocab_path, 'rb') as f:
    SRC.vocab = pickle.load(f)
with open(trg_vocab_path, 'rb') as f:
    TRG.vocab = pickle.load(f)

# Model parameters
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
max_length = 102

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = seq2seq_attention.Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, max_length)
dec = seq2seq_attention.Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device, max_length)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
model = seq2seq_attention.Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

# Load pre-trained weights
model.load_state_dict(torch.load('/home/wei/Desktop/Similarity/self_seq2seq_try/ckpts/chembl33/shuffled_filtered_stereo/att_130eps.pt'))
model.eval()

test_trg = '/home/wei/Desktop/Similarity/chembl33_dataset/curriculum_learning/lev_5*ABpBAp_dataset/tvt/test.trg'
with open(test_trg, 'r') as file:
    contents = file.read()
    trg_smis = contents.split('\n')

test_src = '/home/wei/Desktop/Similarity/chembl33_dataset/curriculum_learning/lev_5*ABpBAp_dataset/tvt/test.src'
with open(test_src, 'r') as file:
    contents = file.read()
    src_smis = contents.split('\n')

tokenizer = vocabulary.SMILESTokenizer()
generated_smiles = multi_gen_test.get_sim_smiles_decoding(
    src_smis[0],
    SRC,
    TRG,
    model,
    device,
    max_length,
    10,
    1.2,  # Temperature for sampling (not relevant for this decoder)
    tokenizer,
    decoder_type=1,  # Standard beam search
    use_masking=True,
    prefix_length=0   # Specify where modifications start
)
print(generated_smiles[0][0])