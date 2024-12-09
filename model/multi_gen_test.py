import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from queue import PriorityQueue
from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors
from rdkit import RDLogger
import heapq

# Suppress RDKit warningsa
RDLogger.DisableLog('rdApp.*')

def subsequent_mask(size):
    """
    Mask out subsequent positions to prevent attention to future tokens.

    Args:
        size (int): The length of the sequence.

    Returns:
        torch.Tensor: A square mask matrix with shape [size, size], where
                      positions that can attend are True, and others are False.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  # Upper triangular matrix
    return torch.from_numpy(subsequent_mask) == 0  # Convert to boolean mask

def get_sim_smiles_decoding(
    smiles,
    src_field,
    trg_field,
    model,
    device,
    max_len,
    beam_width,
    temperature,
    tokenizer,
    decoder_type=0,
    use_masking=True,  # Control masking
    prefix_length=0  # Specify the starting position for modifications
):
    """Generate SMILES strings using different decoding methods.

    Args:
        smiles (str): The input SMILES string.
        src_field: Source field containing vocabulary and tokenization methods.
        trg_field: Target field containing vocabulary and tokenization methods.
        model: The seq2seq model with encoder and decoder.
        device: The device (CPU or GPU) for computation.
        max_len (int): Maximum length of the output sequence.
        beam_width (int): Beam width for beam search.
        temperature (float): Temperature for sampling diversity.
        tokenizer: The tokenizer object with an untokenize method.
        decoder_type (int): Decoder type (0 for standard beam search, 1 for best-first, 2 for sampling).
        use_masking (bool): Whether to apply subsequent masking during decoding.
        prefix_length (int): Number of tokens from the source to use as a prefix after <sos>.

    Returns:
        list of tuples: List containing tuples of untokenized SMILES and their probabilities.
    """
    with torch.no_grad():
        # Tokenize the input SMILES string
        tokens = src_field.tokenize(smiles)
        tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        src_indexes = [src_field.vocab.stoi[token] for token in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src_tensor)
        
        # Select decoding method based on decoder_type
        if decoder_type == 0:
            # Standard beam search decoding
            final_sequences, total_generated_sequences = beam_search_decode(
                model=model, 
                src_tensor=src_tensor, 
                src_mask=src_mask, 
                trg_field=trg_field, 
                beam_width=beam_width, 
                max_len=max_len, 
                device=device, 
                use_masking=use_masking, 
                prefix_length=prefix_length  # Pass the prefix length
            )
        elif decoder_type == 1:
            # Best-first beam search decoding with prefix
            final_sequences, total_generated_sequences = best_first_beam_search_decode(
                model=model, 
                src_tensor=src_tensor, 
                src_mask=src_mask, 
                trg_field=trg_field, 
                beam_width=beam_width, 
                max_len=max_len, 
                device=device, 
                queue_limit=100000, 
                use_masking=use_masking, 
                prefix_length=prefix_length  # Pass the prefix length
            )
        elif decoder_type == 2:
            # Sampling-based decoding with prefix
            final_sequences = sampling_decoder(
                model=model, 
                src_tensor=src_tensor, 
                src_mask=src_mask, 
                trg_field=trg_field, 
                max_len=max_len, 
                temperature=temperature, 
                num_sequences=beam_width,  # Beam width determines the number of sequences
                device=device, 
                use_masking=use_masking, 
                seed=42,  # Fixed seed for reproducibility
                prefix_length=prefix_length  # Pass the prefix length
            )
        else:
            raise ValueError("Invalid decoder_type. Use 0 for standard beam search, 1 for best-first beam search, 2 for sampling.")

        # Untokenize generated sequences
        results = []
        for indexes, prob in final_sequences:
            # Convert token indices to SMILES tokens
            trg_tokens = [trg_field.vocab.itos[i] for i in indexes]
            # Remove <sos> and <eos> tokens
            untokenized_smiles = tokenizer.untokenize(trg_tokens[1:-1])  # Exclude <sos> and <eos>
            results.append((untokenized_smiles, prob))
        
        # Sort results by probability if needed
        if decoder_type != 2:
            results.sort(key=lambda x: x[1], reverse=True)
            top_results = results[:beam_width]
        else:
            top_results = results  # For sampling, return all sampled sequences

    return top_results  # Return the top SMILES and their probabilities

def process_generated_smiles(generated_smiles):
    """
    Process a list of untokenized SMILES, count invalids, duplicates, 
    and return a list of valid and unique molecules.

    Args:
        generated_smiles (list): A list of untokenized SMILES strings.

    Returns:
        tuple: A tuple containing:
            - valid_molecules (list): A list of valid and unique RDKit molecules.
            - invalid_count (int): The number of invalid SMILES.
            - duplicate_count (int): The number of duplicate molecules.
    """
    unique_inchi_keys = set()  # Track unique molecules using InChIKey
    valid_molecules = []  # List of valid and unique RDKit molecules
    invalid_count = 0  # Count of invalid SMILES
    duplicate_count = 0  # Count of duplicate molecules

    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to RDKit molecule

        if mol is None:
            # Increment invalid count for unparseable SMILES
            invalid_count += 1
            continue

        inchi_key = Chem.MolToInchiKey(mol)  # Generate InChIKey for uniqueness

        if inchi_key in unique_inchi_keys:
            # Increment duplicate count if molecule is already encountered
            duplicate_count += 1
            continue

        # Add the molecule to the list and mark it as unique
        unique_inchi_keys.add(inchi_key)
        valid_molecules.append(mol)

    return valid_molecules, invalid_count, duplicate_count

def generate_lots_of_smiles(smi, N):
    """Generate N variants of a given SMILES string by shuffling atom indices."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return set()  # Return an empty set if the molecule is invalid
    indices = [atom.GetIdx() for atom in mol.GetAtoms()]
    lots = set()
    for i in range(N):
        random.shuffle(indices)
        new_mol = Chem.rdmolops.RenumberAtoms(mol, indices)
        new_smi = Chem.MolToSmiles(new_mol, canonical=False)
        lots.add(new_smi)
    return lots

def beam_search_decode(
    model, 
    src_tensor, 
    src_mask, 
    trg_field, 
    beam_width, 
    max_len, 
    device, 
    use_masking=False, 
    prefix_length=0
):
    """
    Performs beam search decoding with a fixed prefix based on `prefix_length`.

    Args:
        model: The seq2seq model with encoder and decoder.
        src_tensor: The source input tensor.
        src_mask: Mask for the source sequence.
        trg_field: Target field containing vocabulary and tokenization methods.
        beam_width: Desired beam width for decoding.
        max_len: Maximum length of the output sequence (including the prefix).
        device: The device (CPU or GPU) for computation.
        use_masking (bool): Whether to apply subsequent masking during decoding.
        prefix_length (int): Number of tokens from the source to use as a prefix after <sos>.

    Returns:
        list of tuples: Generated sequences and their probabilities.
    """
    model.eval()
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

        # Extract prefix: Always include <sos> + first `prefix_length` tokens
        src_indexes = src_tensor.squeeze(0).tolist()  # Convert source tensor to list
        prefix = src_indexes[:1 + prefix_length]  # <sos> + `prefix_length` tokens

        # Initialize sequences and probabilities
        trg_indexes = [prefix]  # Start with the prefix
        trg_probs = [0.0]  # Initial sequence probability

        final_sequences = []  # To store completed sequences (those with EOS)
        total_generated_sequences = 0  # Counter for total sequences generated

        # Adjust max length to account for the prefix length
        tokens_to_generate = max_len - len(prefix)

        for _ in range(tokens_to_generate):  # Generate only up to the adjusted max length
            candidates = []
            for i, indexes in enumerate(trg_indexes):
                # Check if the sequence has reached EOS
                if indexes[-1] == trg_field.vocab.stoi[trg_field.eos_token]:
                    final_sequences.append((indexes, trg_probs[i]))
                    continue  # Skip further expansion for completed sequences

                # Expand the sequence
                trg_tensor = torch.LongTensor(indexes).unsqueeze(0).to(device)
                
                # Apply masking if use_masking is True
                if use_masking:
                    trg_mask = model.make_trg_mask(trg_tensor) * subsequent_mask(len(indexes)).to(device)
                else:
                    trg_mask = model.make_trg_mask(trg_tensor)

                output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                probs = F.softmax(output, dim=-1).squeeze(0)[-1].cpu().numpy()
                vocab_size = len(probs)

                # Expand all possible sequences if beam width exceeds vocab size
                expansion_width = min(beam_width, vocab_size)
                topk_indexes = probs.argsort()[-expansion_width:][::-1]  # Select top-k indices

                for index in topk_indexes:
                    candidate = indexes + [index]
                    candidate_prob = trg_probs[i] + np.log(probs[index].item())
                    candidates.append((candidate, candidate_prob))

                total_generated_sequences += len(topk_indexes)

            # Sort candidates by probability and prune to beam width
            candidates.sort(key=lambda x: x[1], reverse=True)

            if len(candidates) > beam_width:
                trg_indexes = [candidate[0] for candidate in candidates[:beam_width]]
                trg_probs = [candidate[1] for candidate in candidates[:beam_width]]
            else:
                trg_indexes = [candidate[0] for candidate in candidates]
                trg_probs = [candidate[1] for candidate in candidates]

            # Handle sequences that reach EOS during expansion
            finished_sequences = []
            for indexes, prob in zip(trg_indexes, trg_probs):
                if indexes[-1] == trg_field.vocab.stoi[trg_field.eos_token]:
                    final_sequences.append((indexes, prob))
                    finished_sequences.append(indexes)

            # Remove EOS sequences from further processing
            trg_indexes = [indexes for indexes in trg_indexes if indexes not in finished_sequences]
            trg_probs = [prob for indexes, prob in zip(trg_indexes, trg_probs) if indexes not in finished_sequences]

            # If no sequences remain or beam width becomes 0, break
            if not trg_indexes:
                break

        # Add remaining incomplete sequences to the final pool
        final_sequences.extend(zip(trg_indexes, trg_probs))

        # Sort final sequences by probability and prune to desired beam width
        final_sequences.sort(key=lambda x: x[1], reverse=True)
        final_sequences = final_sequences[:beam_width]

        return final_sequences, total_generated_sequences

def best_first_beam_search_decode(
    model, 
    src_tensor, 
    src_mask, 
    trg_field, 
    beam_width, 
    max_len, 
    device, 
    queue_limit=100000, 
    use_masking=False, 
    prefix_length=0
):
    """
    Performs best-first beam search decoding with a fixed prefix based on the `prefix_length`.

    Args:
        model: The seq2seq model with encoder and decoder.
        src_tensor: The source input tensor.
        src_mask: Mask for the source sequence.
        trg_field: Target field containing vocabulary and tokenization methods.
        beam_width: Desired beam width for decoding.
        max_len: Maximum length of the output sequence (including the prefix).
        device: The device (CPU or GPU) for computation.
        queue_limit: Maximum size of the queue for managing sequences.
        use_masking (bool): Whether to apply subsequent masking during decoding.
        prefix_length (int): Number of tokens from the source to use as a prefix after <sos>.

    Returns:
        list of tuples: Generated sequences and their probabilities.
    """
    model.eval()
    with torch.no_grad():
        # Compute encoder output
        enc_src = model.encoder(src_tensor, src_mask)

        # Extract prefix: Always include <sos> + first `prefix_length` tokens
        src_indexes = src_tensor.squeeze(0).tolist()  # Convert source tensor to list
        prefix = src_indexes[:1 + prefix_length]  # <sos> + `prefix_length` tokens
        initial_prob = 0.0  # Log probability of the initial sequence
        sequence_counter = 0  # To keep track of sequence ordering

        # Priority queue to store sequences with cumulative log probabilities
        queue = []
        heapq.heappush(queue, (-initial_prob, sequence_counter, prefix))  # Start with the prefix

        final_sequences = []  # List to store completed sequences
        total_generated_sequences = 0  # Counter for total sequences generated

        while len(final_sequences) < beam_width:
            if not queue:
                break  # No more sequences to process

            # Get the sequence with the highest probability (smallest negative log prob)
            current_neg_prob, _, current_seq = heapq.heappop(queue)
            current_prob = -current_neg_prob  # Convert back to positive log probability
            
            last_token = current_seq[-1]
            
            # Check if the sequence is completed
            if (last_token == trg_field.vocab.stoi[trg_field.eos_token]) or (len(current_seq) >= max_len):
                final_sequences.append((current_seq, current_prob))
                continue  # Move on to the next sequence

            # Prepare the input for the decoder
            trg_tensor = torch.LongTensor(current_seq).unsqueeze(0).to(device)
            
            # Apply masking if use_masking is True
            if use_masking:
                trg_mask = model.make_trg_mask(trg_tensor) * subsequent_mask(len(current_seq)).to(device)
            else:
                trg_mask = model.make_trg_mask(trg_tensor).to(device)
            
            # Get the output probabilities from the model
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            output = output[:, -1, :]  # Get the logits for the last time step
            probs = F.log_softmax(output, dim=-1).squeeze(0)  # Shape: [vocab_size]
            
            # Convert to numpy array and move to CPU
            probs = probs.cpu().numpy()
            
            # Generate all possible next tokens
            vocab_size = probs.shape[0]
            for idx in range(vocab_size):
                token_prob = probs[idx]
                new_seq = current_seq + [idx]
                new_prob = current_prob + token_prob  # Cumulative log probability
                sequence_counter += 1
                new_neg_prob = -new_prob  # Use negative log probability for max-heap
                
                if len(queue) < queue_limit:
                    # Add the new sequence if the queue is not full
                    heapq.heappush(queue, (new_neg_prob, sequence_counter, new_seq))
                elif queue:  # Ensure the queue is not empty
                    # If the queue is full, compare with the worst sequence
                    if new_neg_prob < queue[0][0]:  # Check against the largest negative log prob
                        heapq.heapreplace(queue, (new_neg_prob, sequence_counter, new_seq))

                total_generated_sequences += 1  # Increment sequence count
        
        # Once completed sequences reach beam width, return them
        # Sort final_sequences by cumulative log probabilities
        final_sequences.sort(key=lambda x: x[1], reverse=True)
        
        return final_sequences, total_generated_sequences

def sampling_decoder(
    model, src_tensor, src_mask, trg_field, max_len, temperature, num_sequences, device, use_masking=False, seed=42, prefix_length=0
):
    """
    Performs sampling-based decoding with temperature scaling for diverse SMILES generation,
    while considering a fixed prefix.

    Args:
        model: The seq2seq model with encoder and decoder.
        src_tensor: Source tensor (input sequence).
        src_mask: Mask for the source sequence.
        trg_field: Target field containing vocabulary and tokenization methods.
        max_len (int): Maximum length of the output sequence (including the prefix).
        temperature (float): Temperature for controlling diversity in sampling.
        num_sequences (int): Number of sequences to generate.
        device: The device (CPU or GPU) for computation.
        use_masking (bool): Whether to apply subsequent masking during decoding.
        seed (int): Random seed for reproducibility in token selection.
        prefix_length (int): Number of tokens from the source to use as a prefix after <sos>.

    Returns:
        list of tuples: Generated sequences and their probabilities.
    """
    model.eval()
    all_sequences = []

    # Set the random seed for reproducibility
    np.random.seed(seed)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

        # Extract prefix: Always include <sos> + first `prefix_length` tokens
        src_indexes = src_tensor.squeeze(0).tolist()  # Convert source tensor to list
        prefix = src_indexes[:1 + prefix_length]  # <sos> + `prefix_length` tokens

        # Adjust max length to account for the prefix
        tokens_to_generate = max_len - len(prefix)

        for _ in range(num_sequences):
            trg_indexes = prefix[:]  # Start with the prefix
            trg_probs = [0.0]  # Initialize cumulative probability

            for _ in range(tokens_to_generate):
                trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
                
                # Apply masking if use_masking is True
                if use_masking:
                    trg_mask = model.make_trg_mask(trg_tensor) * subsequent_mask(len(trg_indexes)).to(device)
                else:
                    trg_mask = model.make_trg_mask(trg_tensor)

                # Get output logits for the next token
                output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

                # Apply temperature scaling
                logits = output.squeeze(0)[-1] / temperature

                # Convert logits to probabilities
                probabilities = F.softmax(logits, dim=-1).cpu().numpy()

                # Sample the next token based on probabilities (with fixed seed)
                sampled_index = np.random.choice(len(probabilities), p=probabilities)

                # Append sampled token and its log probability
                trg_indexes.append(sampled_index)
                trg_probs.append(np.log(probabilities[sampled_index]))

                # Stop if <eos> is sampled
                if sampled_index == trg_field.vocab.stoi[trg_field.eos_token]:
                    break

            # Add the sequence and cumulative log probability to the list
            all_sequences.append((trg_indexes, sum(trg_probs)))

    return all_sequences  # Return all sampled SMILES sequences and their log probabilities

def generation_with_variants(
    src_smiles,
    src_field,
    trg_field,
    model,
    device,
    max_len,
    beam_width,
    temperature,
    tokenizer,
    variant_count=10,
    decoder_type=0,
    use_masking=False,
    prefix_length=0  # Specify the starting position for modifications
):
    """
    Generate SMILES for source molecule and its variants, using beam search.

    Args:
        src_smiles (str): The input source SMILES string.
        src_field: Source field containing vocabulary and tokenization methods.
        trg_field: Target field containing vocabulary and tokenization methods.
        model: The seq2seq model with encoder and decoder.
        device: The device (CPU or GPU) for computation.
        max_len (int): Maximum length of the output sequence.
        beam_width (int): Beam width for the beam search.
        temperature (float): Temperature for sampling diversity.
        tokenizer: The tokenizer object with an untokenize method.
        variant_count (int): Number of variants to generate for the source SMILES.
        decoder_type (int): Decoder type (0 for standard beam search, 1 for best-first beam search, 2 for sampling).
        use_masking (bool): Whether to apply masking during decoding.
        prefix_length (int): Number of tokens from the source to use as a prefix after <sos>.

    Returns:
        list: A list of tuples containing generated SMILES and their probabilities.
    """
    all_generated_smiles_and_probs = []

    # Step 1: Generate variants of the source SMILES
    variants = generate_lots_of_smiles(src_smiles, variant_count)
    variants.add(src_smiles)  # Ensure the original SMILES is included

    # Step 2: For each variant, run the chosen decoder and collect SMILES
    for variant in variants:
        generated_smiles = get_sim_smiles_decoding(
            smiles=variant,
            src_field=src_field,
            trg_field=trg_field,
            model=model,
            device=device,
            max_len=max_len,
            beam_width=beam_width,
            temperature=temperature,
            tokenizer=tokenizer,
            decoder_type=decoder_type,
            use_masking=use_masking,
            prefix_length=prefix_length  # Pass the prefix length to the decoding function
        )

        # Add untokenized SMILES and probabilities to the list
        all_generated_smiles_and_probs.extend(generated_smiles)

    return all_generated_smiles_and_probs

def recursive_generation_with_beam(
    src_smiles,
    src_field,
    trg_field,
    model,
    device,
    max_len,
    beam_width,
    steps,
    tokenizer,  # User-provided tokenizer
    temperature=1.0,
    decoder_type=0,
    use_masking=False,
    prefix_length=0
):
    """
    Recursively generate SMILES with beam search across multiple steps.

    Args:
        src_smiles (str): The input source SMILES string.
        src_field: Source field containing vocabulary and tokenization methods.
        trg_field: Target field containing vocabulary and tokenization methods.
        model: The seq2seq model with encoder and decoder.
        device: The device (CPU or GPU) for computation.
        max_len (int): Maximum length of the output sequence.
        beam_width (int): Number of SMILES to generate for each input.
        steps (int): Number of recursive steps for generation.
        tokenizer: The tokenizer object with an untokenize method (must be provided by the user).
        temperature (float): Temperature for sampling diversity.
        decoder_type (int): Decoder type (0 for standard beam search, 1 for best-first beam search, 2 for sampling).
        use_masking (bool): Whether to apply masking during decoding.
        prefix_length (int): Number of tokens from the source to use as a prefix after <sos>.

    Returns:
        list: A list of tuples containing all unique generated SMILES and their probabilities.
    """
    all_generated_smiles = set()  # To ensure uniqueness
    final_smiles_and_probs = []   # To store results with probabilities
    current_smiles_batch = [src_smiles]  # Start with the user-provided SMILES

    for step in range(steps):
        next_smiles_batch = []  # To store SMILES generated in this step
        print(f"Step {step + 1}/{steps}: Processing {len(current_smiles_batch)} SMILES")

        for smiles in current_smiles_batch:
            # Step 1: Use the decoder to generate beam_width SMILES for each input
            generated_smiles = get_sim_smiles_decoding(
                smiles=smiles,
                src_field=src_field,
                trg_field=trg_field,
                model=model,
                device=device,
                max_len=max_len,
                beam_width=beam_width,
                temperature=temperature,
                tokenizer=tokenizer,  # Use the user-provided tokenizer
                decoder_type=decoder_type,
                use_masking=use_masking,
                prefix_length=prefix_length
            )

            # Step 2: Process the generated SMILES and probabilities
            for untokenized_smi, prob in generated_smiles:
                if untokenized_smi not in all_generated_smiles:
                    all_generated_smiles.add(untokenized_smi)
                    final_smiles_and_probs.append((untokenized_smi, prob))
                    next_smiles_batch.append(untokenized_smi)

        # Update the batch for the next step
        current_smiles_batch = next_smiles_batch
        if not current_smiles_batch:  # Stop if no new SMILES were generated
            print("No new SMILES generated; stopping early.")
            break

    return final_smiles_and_probs
