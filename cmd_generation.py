# Standard Libraries
import random
import math
import time
from collections import Counter, defaultdict
import argparse

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

def main(args):
    """
    Main function to set up and generate SMILES using the seq2seq model.

    Args:
        args (Namespace): Parsed command-line arguments.
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
    with open(args.vocab_path, 'rb') as f:
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
    model.load_state_dict(torch.load(args.model_checkpoint_path))
    model.eval()

    tokenizer = vocabulary.SMILESTokenizer()

    if args.generation_method == "generation-normal":
        # Function 1: Generate one or more molecules using decoders
        generated_smiles = multi_gen_test.get_sim_smiles_decoding(
            args.src_smiles,
            SRC,
            TRG,
            model,
            device,
            max_length,
            args.beam_width,
            1.2,  # Temperature for sampling (not relevant for this decoder)
            tokenizer,
            decoder_type=args.decoder_type,
            use_masking=True,
            prefix_length=args.prefix_length
        )
        data = [(smi, prob) for smi, prob in generated_smiles]

    elif args.generation_method == "generation-variant":
        # Function 2: Generate multiple SMILES using variants
        generation_w_var = multi_gen_test.generation_with_variants(
            args.src_smiles,
            SRC,
            TRG,
            model,
            device,
            max_length,
            args.beam_width,
            1.2,
            tokenizer,
            variant_count=10,
            decoder_type=args.decoder_type,
            use_masking=True,
            prefix_length=args.prefix_length
        )
        data = [(smi, prob) for smi, prob in generation_w_var]

    elif args.generation_method == "generation-recursive":
        # Function 3: Generate more SMILES by re-inputting generated SMILES back to the model
        generation_w_recur = multi_gen_test.recursive_generation_with_beam(
            args.src_smiles,
            SRC,
            TRG,
            model,
            device,
            max_length,
            args.beam_width,
            2,  # Number of recursive steps
            tokenizer,  # User-provided tokenizer
            temperature=1.2,
            decoder_type=args.decoder_type,
            use_masking=True,
            prefix_length=args.prefix_length
        )
        data = [(smi, prob) for smi, prob in generation_w_recur]

    else:
        print("Invalid generation method. Please choose from: 'generation-normal', 'generation-variant', 'generation-recursive'.")
        return

    # Create a DataFrame and sort by probability
    df = pd.DataFrame(data, columns=["Generated_SMILES", "Probability"])
    df = df.sort_values(by="Probability", ascending=False)

    # Save to CSV
    df.to_csv(args.output_path, index=False)
    print(f"Generated SMILES saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMILES Generation Script")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--decoder_type", type=int, required=True, help="Decoder type (e.g., 0 for standard beam search).")
    parser.add_argument("--prefix_length", type=int, required=True, help="Prefix length.")
    parser.add_argument("--beam_width", type=int, required=True, help="Beam width for SMILES generation.")
    parser.add_argument("--generation_method", type=str, required=True, choices=["generation-normal", "generation-variant", "generation-recursive"], help="Generation method to use.")
    parser.add_argument("--src_smiles", type=str, required=True, help="Source SMILES string to generate from.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated SMILES CSV file.")

    args = parser.parse_args()

    main(args)
