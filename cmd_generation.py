import argparse
import torch
from torchtext.legacy.data import Field
from models import vocabulary, seq2seq_attention, multi_gen_real_final
import pickle
import pandas as pd

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

    # Read input SMILES (single or from file)
    input_smiles_list = []
    if args.src_smiles.endswith('.smi'):
        # Read SMILES from file
        with open(args.src_smiles, 'r') as f:
            input_smiles_list = [line.strip() for line in f if line.strip()]
    else:
        # Single SMILES input
        input_smiles_list = [args.src_smiles]

    all_data = []  # To collect all generated SMILES and probabilities

    for input_smiles in input_smiles_list:
        if args.input_selection == "normal":
            # Generate SMILES using normal method
            generated_smiles = multi_gen_real_final.get_sim_smiles_decoding(
                input_smiles,
                SRC,
                TRG,
                model,
                device,
                max_length,
                args.beam_width,
                1.2,  # Temperature for sampling (not relevant for this decoder)
                generation_method=args.generation_method,
                use_masking=True,
                prefix_length=args.prefix_length,
                invalid_check=args.invalid_check
            )
        elif args.input_selection == "variant":
            # Generate SMILES using variants method
            generated_smiles = multi_gen_real_final.generation_with_variants(
                input_smiles,
                SRC,
                TRG,
                model,
                device,
                max_length,
                args.beam_width,
                1.2,
                variant_count=10,
                generation_method=args.generation_method,
                use_masking=True,
                prefix_length=args.prefix_length,
                invalid_check=args.invalid_check
            )
        elif args.input_selection == "recursive":
            # Generate SMILES using recursive method
            generated_smiles = multi_gen_real_final.recursive_generation_with_beam(
                input_smiles,
                SRC,
                TRG,
                model,
                device,
                max_length,
                args.beam_width,
                2,  # Number of recursive steps
                temperature=1.2,
                generation_method=args.generation_method,
                use_masking=True,
                prefix_length=args.prefix_length,
                invalid_check=args.invalid_check
            )
        else:
            print("Invalid generation method. Skipping SMILES.")
            continue

        # Add input SMILES to each generated SMILES
        for smi, prob in generated_smiles:
            all_data.append((input_smiles, smi, prob))

    # Create a DataFrame
    df = pd.DataFrame(all_data, columns=["Input_SMILES", "Generated_SMILES", "Probability"])

    # Sort probabilities within each input SMILES cluster
    df = df.sort_values(by=["Input_SMILES", "Probability"], ascending=[True, False])

    # Save to CSV
    df.to_csv(args.output_path, index=False)
    print(f"Generated SMILES saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMILES Generation Script")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--generation_method", type=str, required=True, help="Generation method (e.g., 'BS' for standard beam search, 'BFBS' for best-first beam search, 'SD' for sampling decoder).")
    parser.add_argument("--prefix_length", type=int, required=True, help="Prefix length.")
    parser.add_argument("--invalid_check", type=bool, required=True, help="do invalid check or not.")
    parser.add_argument("--beam_width", type=int, required=True, help="Beam width for SMILES generation.")
    parser.add_argument("--input_selection", type=str, required=True, choices=["normal", "variant", "recursive"], help="How to modify input SMILES.")
    parser.add_argument("--src_smiles", type=str, required=True, help="Source SMILES string or path to a .smi file containing SMILES.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated SMILES CSV file.")

    args = parser.parse_args()

    main(args)
