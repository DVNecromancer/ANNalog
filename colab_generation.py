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

def main(vocab_path, model_checkpoint_path, decoder_type, prefix_length, beam_width, generation_method, src_smiles):
    """
    Main function to set up and generate SMILES using the seq2seq model.

    Args:
        vocab_path (str): Path to the shared vocabulary pickle file.
        model_checkpoint_path (str): Path to the model checkpoint.
        decoder_type (int): Decoder type (e.g., 0 for standard beam search).
        prefix_length (int): Number of tokens to use as a prefix after <sos>.
        beam_width (int): Beam width for SMILES generation.
        generation_method (str): The generation method ("generation-normal", "generation-variant", "generation-recursive").
        src_smiles (str): The source SMILES string to generate from.
    """
    # Set up SRC and TRG fields
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
    with open(vocab_path, 'rb') as f:
        shared_vocab = pickle.load(f)
    SRC.vocab = shared_vocab
    TRG.vocab = shared_vocab

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

    # Set up the model
    enc = seq2seq_attention.Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, max_length)
    dec = seq2seq_attention.Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device, max_length)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    model = seq2seq_attention.Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    # Load pre-trained weights
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()

    tokenizer = vocabulary.SMILESTokenizer()

    if generation_method == "generation-normal":
        # Function 1: Generate one or more molecules using decoders
        generated_smiles = multi_gen_test.get_sim_smiles_decoding(
            src_smiles,
            SRC,
            TRG,
            model,
            device,
            max_length,
            beam_width,
            1.2,  # Temperature for sampling (not relevant for this decoder)
            tokenizer,
            decoder_type=decoder_type,
            use_masking=True,
            prefix_length=prefix_length
        )
        print("Generated SMILES (Normal):", generated_smiles[0][0])

    elif generation_method == "generation-variant":
        # Function 2: Generate multiple SMILES using variants
        generation_w_var = multi_gen_test.generation_with_variants(
            src_smiles,
            SRC,
            TRG,
            model,
            device,
            max_length,
            beam_width,
            1.2,
            tokenizer,
            variant_count=10,
            decoder_type=decoder_type,
            use_masking=True,
            prefix_length=prefix_length
        )
        print("Generated SMILES with Variants:", generation_w_var[0][0])

    elif generation_method == "generation-recursive":
        # Function 3: Generate more SMILES by re-inputting generated SMILES back to the model
        generation_w_recur = multi_gen_test.recursive_generation_with_beam(
            src_smiles,
            SRC,
            TRG,
            model,
            device,
            max_length,
            beam_width,
            2,  # Number of recursive steps
            tokenizer,  # User-provided tokenizer
            temperature=1.2,
            decoder_type=decoder_type,
            use_masking=True,
            prefix_length=prefix_length
        )
        print("Generated SMILES with Recursion:", generation_w_recur[0][0])

    else:
        print("Invalid generation method. Please choose from: 'generation-normal', 'generation-variant', 'generation-recursive'.")

if __name__ == "__main__":
    vocab_path = input("Enter the path to the vocabulary file: ").strip()
    model_checkpoint_path = input("Enter the path to the model checkpoint file: ").strip()
    decoder_type = int(input("Enter the decoder type (e.g., 0 for standard beam search): ").strip())
    prefix_length = int(input("Enter the prefix length: ").strip())
    beam_width = int(input("Enter the beam width: ").strip())
    generation_method = input("Enter the generation method ('generation-normal', 'generation-variant', 'generation-recursive'): ").strip()
    src_smiles = input("Enter the source SMILES string (in quotes): ").strip()

    main(vocab_path, model_checkpoint_path, decoder_type, prefix_length, beam_width, generation_method, src_smiles)
