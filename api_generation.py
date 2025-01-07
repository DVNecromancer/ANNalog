import argparse
import pandas as pd
from annalog.model_handler import SMILESModelHandler
from annalog.SMILES_generator import SMILESGenerator
import torch

def main(args):
    """
    Main function to set up and generate SMILES using the API.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    # Initialize the model handler and generator
    handler = SMILESModelHandler(
        src_vocab_path=args.vocab_path,
        trg_vocab_path=args.vocab_path,
        model_path=args.model_checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    generator = SMILESGenerator(handler)

    # Read input SMILES (single or from file)
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
            generated_smiles = generator.generate_smiles(
                input_smiles=input_smiles,
                beam_width=args.beam_width,
                generation_method=args.generation_method,
                prefix_length=args.prefix_length,
                use_masking=True,
                invalid_check=args.invalid_check
            )
        elif args.input_selection == "variant":
            # Generate SMILES using variants method
            generated_smiles = generator.generation_with_variants(
                input_smiles=input_smiles,
                beam_width=args.beam_width,
                variant_count=10,
                generation_method=args.generation_method,
                prefix_length=args.prefix_length,
                use_masking=True,
                invalid_check=args.invalid_check
            )
        elif args.input_selection == "recursive":
            # Generate SMILES using recursive method
            generated_smiles = generator.recursive_generation(
                input_smiles=input_smiles,
                beam_width=args.beam_width,
                steps=2,  # Number of recursive steps
                generation_method=args.generation_method,
                prefix_length=args.prefix_length,
                use_masking=True,
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
    parser = argparse.ArgumentParser(description="SMILES Generation Script with API")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--generation_method", type=str, required=True, help="Generation method (e.g., 'BS' for standard beam search, 'BFBS' for best-first beam search, 'SD' for sampling decoder).")
    parser.add_argument("--prefix_length", type=int, required=True, help="Prefix length.")
    parser.add_argument("--invalid_check", type=bool, required=True, help="Do invalid check or not.")
    parser.add_argument("--beam_width", type=int, required=True, help="Beam width for SMILES generation.")
    parser.add_argument("--input_selection", type=str, required=True, choices=["normal", "variant", "recursive"], help="How to modify input SMILES.")
    parser.add_argument("--src_smiles", type=str, required=True, help="Source SMILES string or path to a .smi file containing SMILES.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated SMILES CSV file.")

    args = parser.parse_args()

    main(args)
