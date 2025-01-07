from .model_files import multi_gen_final
from rdkit import Chem

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
        self.SRC, self.TRG, self.model, self.device = model_handler.get_model_and_fields()
        self.max_length = model_handler.max_length

    def generate_smiles(self, input_smiles, beam_width=100, generation_method='BS', prefix_length=0, use_masking=True, invalid_check=False):
        """
        Generate SMILES using the model.

        Args:
            input_smiles (str): Input SMILES string.
            beam_width (int): Beam width for beam search.
            generation_method (str): Generation method ('SD' for standard beam search).
            prefix_length (int): Length of the prefix to preserve in generation.
            use_masking (bool): Whether to use masking during generation.
            invalid_check (bool): Whether to check for invalid SMILES.

        Returns:
            list: Generated SMILES strings.
        """
        return multi_gen_final.get_sim_smiles_decoding(
            smiles=input_smiles,
            src_field=self.SRC,
            trg_field=self.TRG,
            model=self.model,
            device=self.device,
            max_len=self.max_length,
            beam_width=beam_width,
            temperature=1.2,  # Default temperature
            generation_method=generation_method,
            use_masking=use_masking,
            prefix_length=prefix_length,
            invalid_check=invalid_check
        )

    def generation_with_variants(self, input_smiles, beam_width=100, variant_count=10, generation_method='BS', prefix_length=0, use_masking=True,invalid_check=False):
        """
        Generate SMILES for source molecule and its variants, using different generation methods.
        """
        all_generated_smiles_and_probs = []
        unique_smiles = set()

        # Generate variants of the source SMILES
        variants = multi_gen_final.generate_lots_of_smiles(input_smiles, variant_count)
        variants.add(input_smiles)  # Ensure the original SMILES is included

        for variant in variants:
            generated_smiles = multi_gen_final.get_sim_smiles_decoding(
                smiles=variant,
                src_field=self.SRC,
                trg_field=self.TRG,
                model=self.model,
                device=self.device,
                max_len=self.max_length,
                beam_width=beam_width,
                temperature=1.2,
                generation_method=generation_method,
                use_masking=use_masking,
                prefix_length=prefix_length,
                invalid_check=invalid_check
            )

            for smi, prob in generated_smiles:
                if smi not in unique_smiles:
                    unique_smiles.add(smi)
                    all_generated_smiles_and_probs.append((smi, prob))

        return all_generated_smiles_and_probs

    def recursive_generation(self, input_smiles, beam_width=100, steps=3, generation_method='BS', prefix_length=0, use_masking=False,invalid_check=False):
        """
        Recursively generate SMILES with different generation methods.
        """
        all_generated_smiles = set()
        final_smiles_and_probs = []
        current_smiles_batch = [(input_smiles, 0.0)]  # Start with the initial SMILES and a base probability of 0.0

        for step in range(steps):
            print(f"Step {step + 1}/{steps}: Generating SMILES")
            next_smiles_batch = []

            for smiles, cumulative_prob in current_smiles_batch:
                generated = multi_gen_final.get_sim_smiles_decoding(
                    smiles=smiles,
                    src_field=self.SRC,
                    trg_field=self.TRG,
                    model=self.model,
                    device=self.device,
                    max_len=self.max_length,
                    beam_width=beam_width,
                    temperature=1.2,
                    generation_method=generation_method,
                    use_masking=use_masking,
                    prefix_length=prefix_length,
                    invalid_check=invalid_check
                )

                for untokenized_smi, prob in generated:
                    new_cumulative_prob = cumulative_prob + prob
                    if untokenized_smi not in all_generated_smiles:
                        all_generated_smiles.add(untokenized_smi)
                        final_smiles_and_probs.append((untokenized_smi, new_cumulative_prob))
                        next_smiles_batch.append((untokenized_smi, new_cumulative_prob))

            if not next_smiles_batch:
                print("No new SMILES generated; stopping early.")
                break

            current_smiles_batch = next_smiles_batch

        return final_smiles_and_probs