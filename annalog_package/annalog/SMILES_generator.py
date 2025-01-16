from .model_files import multi_gen_final
from .model_files import vocabulary
from rdkit import Chem

tokenizer = vocabulary.SMILESTokenizer()

class SMILESGenerator:
    """
    A class to generate SMILES using the initialized model.
    """
    def __init__(self, model_handler):
        """
        Initialize the SMILES generator.

        Args:
            model_handler (SMILESModelHandler): An instance of the SMILESModelHandler class.
        """
        self.SRC, self.TRG, self.model, self.device, self.use_masking = model_handler.get_model_and_fields()
        self.max_length = model_handler.max_length

    def generate_smiles(self, input_smiles, generation_number=100, generation_method='beam', temperature=1.2, prefix=0, filter_invalid=False):
        """
        Generate SMILES using the model.

        Args:
            input_smiles (str): Input SMILES string.
            beam_width (int): Beam width for beam search.
            generation_method (str): Generation method ('sample' for sampling decoder, 'beam' for best-first beam search).
            temp (float): Temperature for generation.
            prefix (str or int): Prefix for generation, either a string, an integer, or 0 for no fixed prefix.
            filter_invalid (bool): Whether to filter invalid SMILES.

        Returns:
            list: Generated SMILES strings.
        """
        # Validate and process the prefix
        if isinstance(prefix, str):
            # Calculate the length of the prefix string
            prefix_length = len(prefix)

            # Check if the prefix matches the beginning of the input SMILES
            if input_smiles[:prefix_length] == prefix:
                actual_prefix = prefix

                # Tokenize the prefix and determine the token length
                tokenized_prefix = tokenizer.tokenize(actual_prefix)
                token_prefix_length = len(tokenized_prefix)

            else:
                raise ValueError(
                    f"Provided prefix '{prefix}' does not match the beginning of the input SMILES. "
                    f"Please re-run the code with a correct prefix."
                )

        elif isinstance(prefix, int):
            if prefix == 0:  # No fixed prefix
                token_prefix_length = 0
            else:
                # Take the first `prefix` characters of the input SMILES
                prefix_length = prefix
                actual_prefix = input_smiles[:prefix_length]

                # Tokenize the prefix and determine the token length
                tokenized_prefix = tokenizer.tokenize(actual_prefix)
                token_prefix_length = len(tokenized_prefix)
        else:
            raise ValueError("Prefix must be either a string or an integer.")

        # If no fixed prefix, skip tokenizing and set token_prefix_length to 0
        if prefix == 0:
            token_prefix_length = 0

        # Use the token prefix length for model's prefix_length
        return multi_gen_final.get_sim_smiles_decoding(
            smiles=input_smiles,
            src_field=self.SRC,
            trg_field=self.TRG,
            model=self.model,
            device=self.device,
            max_len=self.max_length,
            beam_width=generation_number,
            temperature=temperature,
            generation_method=generation_method,
            use_masking=self.use_masking,
            prefix_length=token_prefix_length,  # Use tokenized prefix length
            filter_invalid=filter_invalid
        )

    def generate_variants(self, input_smiles, num_variants):
        """
        Generate variants of the input SMILES.

        Args:
            input_smiles (str): Input SMILES string.
            num_variants (int): Number of variants to generate.

        Returns:
            list: Generated SMILES variants.
        """
        return multi_gen_final.generate_lots_of_smiles(input_smiles, num_variants)
