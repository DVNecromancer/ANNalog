import argparse
import json
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

    # Single SMILES input
    input_smiles = args.input_SMILES.strip()

    # Generate SMILES
    generated_smiles = generator.generate_smiles(
        input_smiles=input_smiles,
        beam_width=args.beam_width,
        generation_method=args.generation_method,
        prefix=args.prefix,
        filter_invalid=args.filter_invalid
    )

    # Collect results in a variable
    results = [smi for smi, prob in generated_smiles]

    # Print results as JSON
    print(json.dumps(results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMILES Generation Script with API")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--generation_method", type=str, required=True, help="Generation method (e.g., 'beam' for best-first beam search, 'sample' for sampling decoder).")
    parser.add_argument("--prefix", required=True, help="Fixed prefix (can be int or str).")
    parser.add_argument("--filter_invalid", type=bool, required=True, help="Filter out invalid SMILES or not.")
    parser.add_argument("--beam_width", type=int, required=True, help="Beam width for SMILES generation.")
    parser.add_argument("--input_SMILES", type=str, required=True, help="Source SMILES string.")
    
    args = parser.parse_args()

    main(args)
