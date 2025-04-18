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

    def generate_smiles(self, input_smiles, generation_number=100, generation_method='C-beam', temperature=1.2, prefix=0, filter_invalid=False):
        """
        Generate SMILES using the model.

        Args:
            input_smiles (str): Input SMILES string.
            generation_number (int): Beam width or number of generated sequences.
            generation_method (str): Generation method ('sample', 'C-beam','BF-beam' etc.).
            temperature (float): Temperature for generation.
            prefix (str or int): Fixed prefix to use (converted internally to a token count).
            filter_invalid (bool): Whether to filter out invalid SMILES.

        Returns:
            list: Generated SMILES strings with their probabilities.
        """
        # Validate and process the prefix
        if isinstance(prefix, str):
            prefix_length = len(prefix)
            if input_smiles[:prefix_length] == prefix:
                actual_prefix = prefix
                tokenized_prefix = tokenizer.tokenize(actual_prefix)
                token_prefix_length = len(tokenized_prefix)
            else:
                raise ValueError(
                    f"Provided prefix '{prefix}' does not match the beginning of the input SMILES. "
                    f"Please re-run the code with a correct prefix."
                )
        elif isinstance(prefix, int):
            if prefix == 0:
                token_prefix_length = 0
            else:
                prefix_length = prefix
                actual_prefix = input_smiles[:prefix_length]
                tokenized_prefix = tokenizer.tokenize(actual_prefix)
                token_prefix_length = len(tokenized_prefix)
        else:
            raise ValueError("Prefix must be either a string or an integer.")

        if prefix == 0:
            token_prefix_length = 0

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
            prefix_length=token_prefix_length,
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

    def score_smiles(self, input_smiles, target_smiles, prefix=0):
        """
        Compute the Negative Log Likelihood (NLL) for a given target SMILES given an input SMILES.

        This function calls the compute_nll function from the multi_gen_final module.
        The 'prefix' parameter is passed directly (as an integer) to ignore that many tokens (after <sos>)
        during the NLL calculation.

        Args:
            input_smiles (str): The source SMILES string.
            target_smiles (str): The target SMILES string to score.
            prefix (int): Number of tokens (after <sos>) to ignore in the NLL computation.
                          (Prefix processing at the generator level should convert any string prefix to an integer token count.)

        Returns:
            tuple: (total_nll, avg_nll) where total_nll is the sum of negative log probabilities
                   (excluding the fixed prefix tokens) and avg_nll is the average per-token NLL.
        """
        return multi_gen_final.compute_nll(
            input_smiles,
            target_smiles,
            self.SRC,
            self.TRG,
            self.model,
            self.device,
            prefix
        )