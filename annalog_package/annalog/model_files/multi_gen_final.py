import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from queue import PriorityQueue
import heapq
from . import vocabulary
import partialsmiles as ps
from rdkit import Chem

tokenizer = vocabulary.SMILESTokenizer()

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

def validate_smiles(smi, partial=True):
    try:
        if partial:
            ps.ParseSmiles(smi, partial=True)
        else:
            ps.ParseSmiles(smi)
        return True
    except ps.Error:
        return False

def get_sim_smiles_decoding(
    smiles,
    src_field,
    trg_field,
    model,
    device,
    max_len,
    beam_width,
    temperature,
    generation_method='BS',  # Changed from decoder_type to generation_method
    use_masking=True,  # Control masking
    prefix_length=0,  # Specify the starting position for modifications
    invalid_check=False
):
    """
    Generate SMILES strings using different decoding methods.

    Args:
        smiles (str): The input SMILES string.
        src_field: Source field containing vocabulary and tokenization methods.
        trg_field: Target field containing vocabulary and tokenization methods.
        model: The seq2seq model with encoder and decoder.
        device: The device (CPU or GPU) for computation.
        max_len (int): Maximum length of the output sequence.
        beam_width (int): Beam width for beam search.
        temperature (float): Temperature for sampling diversity.
        generation_method (str): Decoding method ('BS' for Beam Search, 'BFBS' for Best-First Beam Search, 'SD' for Sampling Decoder).
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
        
        # Select decoding method based on generation_method
        if generation_method == 'BS':
            # Standard beam search decoding
            final_sequences = beam_search_decode(
                model=model, 
                src_tensor=src_tensor, 
                src_mask=src_mask, 
                trg_field=trg_field, 
                beam_width=beam_width, 
                max_len=max_len, 
                device=device, 
                use_masking=use_masking, 
                prefix_length=prefix_length, 
                invalid_check=invalid_check  # Enable invalid sequence checking
            )
        elif generation_method == 'BFBS':
            # Best-first beam search decoding with prefix
            final_sequences = best_first_beam_search_decode(
                model=model, 
                src_tensor=src_tensor, 
                src_mask=src_mask, 
                trg_field=trg_field, 
                beam_width=beam_width, 
                max_len=max_len, 
                device=device, 
                queue_limit=100000, 
                use_masking=use_masking, 
                prefix_length=prefix_length,  # Pass the prefix length
                invalid_check=invalid_check
            )
        elif generation_method == 'SD':
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
                prefix_length=prefix_length,  # Pass the prefix length
                invalid_check=invalid_check
            )
        else:
            raise ValueError("Invalid generation_method. Use 'BS' for Beam Search, 'BFBS' for Best-First Beam Search, 'SD' for Sampling Decoder.")

        # Untokenize generated sequences
        results = []
        for indexes, prob in final_sequences:
            # Convert token indices to SMILES tokens
            trg_tokens = [trg_field.vocab.itos[i] for i in indexes]
            # Remove <sos> and <eos> tokens
            untokenized_smiles = tokenizer.untokenize(trg_tokens[1:-1])  # Exclude <sos> and <eos>
            results.append((untokenized_smiles, prob))
        
        # Sort results by probability if needed
        if generation_method != 'SD':
            results.sort(key=lambda x: x[1], reverse=True)
            top_results = results[:beam_width]
        else:
            top_results = results  # For sampling, return all sampled sequences

    return top_results  # Return the top SMILES and their probabilities

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
    prefix_length=0, 
    invalid_check=False
):
    """
    Performs beam search decoding with a fixed prefix and handles invalid sequences.

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
        invalid_check (bool): Whether to validate sequences for SMILES correctness.

    Returns:
        list of tuples: Generated sequences and their probabilities.
    """
    model.eval()
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

        # Extract prefix: Always include <sos> + first prefix_length tokens
        src_indexes = src_tensor.squeeze(0).tolist()
        prefix = src_indexes[:1 + prefix_length]

        trg_indexes = [prefix]
        trg_probs = [0.0]
        final_sequences = []

        tokens_to_generate = max_len - len(prefix)

        for _ in range(tokens_to_generate):
            trg_tensors = [torch.LongTensor(indexes).to(device) for indexes in trg_indexes]
            trg_tensors = torch.nn.utils.rnn.pad_sequence(
                trg_tensors, batch_first=True, padding_value=trg_field.vocab.stoi[trg_field.pad_token]
            )

            trg_mask = subsequent_mask(trg_tensors.size(1)).to(device) if use_masking else model.make_trg_mask(trg_tensors)

            output, _ = model.decoder(trg_tensors, enc_src.repeat(len(trg_indexes), 1, 1), trg_mask, src_mask)
            probs = F.softmax(output[:, -1, :], dim=-1)

            expanded_candidates = []
            for i, (indexes, prob_row) in enumerate(zip(trg_indexes, probs)):
                vocab_size = prob_row.size(0)
                topk_probs, topk_indices = prob_row.topk(vocab_size)

                for j in range(vocab_size):
                    candidate = indexes + [topk_indices[j].item()]
                    candidate_prob = trg_probs[i] + torch.log(topk_probs[j] + 1e-10).item()
                    expanded_candidates.append((candidate_prob, candidate))

            expanded_candidates.sort(key=lambda x: x[0], reverse=True)

            # Handle 3x beam width for invalid_check
            if invalid_check:
                num_candidates = min(len(expanded_candidates), 3 * beam_width)
                top_candidates = expanded_candidates[:num_candidates]

                valid_candidates = []
                for prob, candidate in top_candidates:
                    token_sequence = [trg_field.vocab.itos[token] for token in candidate]

                    if "<eos>" in token_sequence:
                        eos_index = token_sequence.index("<eos>")
                        token_sequence = token_sequence[:eos_index]

                    filtered_token_sequence = [
                        token for token in token_sequence if token not in {"<sos>", "<eos>"}
                    ]

                    if any(token in {"<unk>", "<pad>"} for token in filtered_token_sequence):
                        continue

                    partial_smiles = tokenizer.untokenize(filtered_token_sequence)
                    if not validate_smiles(partial_smiles, partial=True):
                        continue

                    valid_candidates.append((prob, candidate))

                trg_indexes = [candidate for _, candidate in valid_candidates]
                trg_probs = [prob for prob, _ in valid_candidates]

                if len(trg_indexes) > 3 * beam_width:
                    trg_indexes = trg_indexes[:3 * beam_width]
                    trg_probs = trg_probs[:3 * beam_width]

            else:
                trg_indexes = [candidate for _, candidate in expanded_candidates[:beam_width]]
                trg_probs = [prob for prob, _ in expanded_candidates[:beam_width]]

            # Process finished sequences
            finished_sequences = []
            for indexes, prob in zip(trg_indexes, trg_probs):
                if indexes[-1] == trg_field.vocab.stoi[trg_field.eos_token]:
                    if invalid_check:
                        # Validate finished sequences if invalid_check is ON
                        token_sequence = [trg_field.vocab.itos[token] for token in indexes]
                        filtered_token_sequence = token_sequence[1:-1]
                        full_smiles = tokenizer.untokenize(filtered_token_sequence)

                        if validate_smiles(full_smiles, partial=False):
                            final_sequences.append((indexes, prob))
                            finished_sequences.append(indexes)
                    else:
                        # Record all finished sequences if invalid_check is OFF
                        final_sequences.append((indexes, prob))
                        finished_sequences.append(indexes)

            # Remove finished sequences from trg_indexes and trg_probs
            trg_indexes = [indexes for indexes in trg_indexes if indexes not in finished_sequences]
            trg_probs = [prob for indexes, prob in zip(trg_indexes, trg_probs) if indexes not in finished_sequences]

            # Terminate early if we have enough completed sequences
            if len(final_sequences) >= beam_width:
                break

        # Sort final sequences and return the top beam_width
        final_sequences.sort(key=lambda x: x[1], reverse=True)
        return final_sequences[:beam_width]

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
    prefix_length=0, 
    invalid_check=False
):
    """
    Performs best-first beam search decoding with a fixed prefix and invalid check.

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
        prefix_length (int): Number of tokens from the source to use as a prefix.
        invalid_check (bool): Whether to validate sequences for SMILES correctness.

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

        # Priority queue to store sequences with cumulative log probabilities
        queue = []
        heapq.heappush(queue, (-initial_prob, prefix))  # Start with the prefix

        final_sequences = []  # List to store completed sequences

        while len(final_sequences) < beam_width:
            if not queue:
                break  # No more sequences to process

            # Get the sequence with the highest probability (smallest negative log prob)
            current_neg_prob, current_seq = heapq.heappop(queue)
            current_prob = -current_neg_prob  # Convert back to positive log probability

            # Skip invalid sequences if invalid_check is enabled
            if invalid_check:
                token_sequence = [trg_field.vocab.itos[token] for token in current_seq]

                # Remove everything after <eos> and remove <sos> and <eos>
                if "<eos>" in token_sequence:
                    eos_index = token_sequence.index("<eos>")
                    token_sequence = token_sequence[:eos_index]  # Keep only up to <eos>
                filtered_token_sequence = [
                    token for token in token_sequence if token not in {"<sos>", "<eos>"}
                ]

                # Skip sequences with <unk> or <pad> between <sos> and <eos>
                if any(token in {"<unk>", "<pad>"} for token in filtered_token_sequence):
                    continue  # Treat as invalid and move to the next sequence

                # Untokenize the filtered sequence to get the partial SMILES
                partial_smiles = tokenizer.untokenize(filtered_token_sequence)

                # Validate the partial SMILES
                if not validate_smiles(partial_smiles, partial=True):
                    continue  # Skip invalid sequences

            last_token = current_seq[-1]

            # Check if the sequence is completed
            if (last_token == trg_field.vocab.stoi[trg_field.eos_token]) or (len(current_seq) >= max_len):
                # Final validity check for full SMILES
                if invalid_check:
                    token_sequence = [trg_field.vocab.itos[token] for token in current_seq]
                    filtered_token_sequence = token_sequence[1:-1]  # Remove <sos> and <eos>
                    full_smiles = tokenizer.untokenize(filtered_token_sequence)

                    if not validate_smiles(full_smiles, partial=False):
                        continue  # Skip invalid full SMILES

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
                new_neg_prob = -new_prob  # Use negative log probability for max-heap

                if len(queue) < queue_limit:
                    # Add the new sequence if the queue is not full
                    heapq.heappush(queue, (new_neg_prob, new_seq))
                elif queue:  # Ensure the queue is not empty
                    # If the queue is full, compare with the worst sequence
                    if new_neg_prob < queue[0][0]:  # Check against the largest negative log prob
                        heapq.heapreplace(queue, (new_neg_prob, new_seq))

        # Once completed sequences reach beam width, return them
        # Sort final_sequences by cumulative log probabilities
        final_sequences.sort(key=lambda x: x[1], reverse=True)

        return final_sequences

def sampling_decoder(
    model, src_tensor, src_mask, trg_field, max_len, temperature, num_sequences, device, use_masking=False, seed=42, prefix_length=0, invalid_check=False
):
    """
    Performs sampling-based decoding with temperature scaling for diverse SMILES generation,
    while considering a fixed prefix and optionally validating both partial and full SMILES.

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
        invalid_check (bool): Whether to validate sequences for SMILES correctness.

    Returns:
        list of tuples: Generated sequences and their probabilities.
    """
    model.eval()
    unique_sequences = set()
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

        while len(all_sequences) < num_sequences:
            remaining_sequences = num_sequences - len(all_sequences)

            for _ in range(remaining_sequences):
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

                    # Resample until a valid token is found
                    for _ in range(len(probabilities)):  # Prevent infinite loops
                        sampled_index = np.random.choice(len(probabilities), p=probabilities)

                        # Create the candidate sequence and validate partial SMILES
                        candidate_indexes = trg_indexes + [sampled_index]

                        if invalid_check:
                            token_sequence = [trg_field.vocab.itos[token] for token in candidate_indexes]

                            # Remove <sos>, <eos>, and everything after <eos> for partial SMILES
                            if "<eos>" in token_sequence:
                                eos_index = token_sequence.index("<eos>")
                                token_sequence = token_sequence[:eos_index]
                            filtered_token_sequence = [
                                token for token in token_sequence if token not in {"<sos>", "<eos>"}
                            ]

                            # Untokenize to get the partial SMILES
                            partial_smiles = tokenizer.untokenize(filtered_token_sequence)

                            if validate_smiles(partial_smiles, partial=True):
                                # Valid partial SMILES; accept this token
                                trg_indexes.append(sampled_index)
                                trg_probs.append(np.log(probabilities[sampled_index]))
                                break
                        else:
                            # No invalid check; accept the token directly
                            trg_indexes.append(sampled_index)
                            trg_probs.append(np.log(probabilities[sampled_index]))
                            break

                    # Stop if <eos> is sampled
                    if sampled_index == trg_field.vocab.stoi[trg_field.eos_token]:
                        break

                # Validate the full SMILES if invalid_check is enabled
                if invalid_check:
                    token_sequence = [trg_field.vocab.itos[token] for token in trg_indexes]
                    filtered_token_sequence = token_sequence[1:-1]  # Remove <sos> and <eos>
                    full_smiles = tokenizer.untokenize(filtered_token_sequence)

                    if not validate_smiles(full_smiles, partial=False):
                        continue  # Skip invalid full SMILES

                # Add the sequence and cumulative log probability to the set for uniqueness
                sequence_tuple = (tuple(trg_indexes), sum(trg_probs))
                if sequence_tuple not in unique_sequences:
                    unique_sequences.add(sequence_tuple)
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
    variant_count=10,
    generation_method='BS',  # Updated from decoder_type
    use_masking=True,
    prefix_length=0,  # Specify the starting position for modifications
    invalid_check=False
):
    """
    Generate SMILES for source molecule and its variants, using different generation methods.

    Args:
        src_smiles (str): The input source SMILES string.
        src_field: Source field containing vocabulary and tokenization methods.
        trg_field: Target field containing vocabulary and tokenization methods.
        model: The seq2seq model with encoder and decoder.
        device: The device (CPU or GPU) for computation.
        max_len (int): Maximum length of the output sequence.
        beam_width (int): Beam width for the beam search.
        temperature (float): Temperature for sampling diversity.
        variant_count (int): Number of variants to generate for the source SMILES.
        generation_method (str): Generation method ('BS', 'BFBS', 'SD').
        use_masking (bool): Whether to apply masking during decoding.
        prefix_length (int): Number of tokens from the source to use as a prefix after <sos>.

    Returns:
        list: A list of tuples containing unique generated SMILES and their probabilities.
    """
    all_generated_smiles_and_probs = []
    unique_smiles = set()

    # Step 1: Generate variants of the source SMILES
    variants = generate_lots_of_smiles(src_smiles, variant_count)
    variants.add(src_smiles)  # Ensure the original SMILES is included

    # Step 2: For each variant, run the chosen generation method and collect SMILES
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
            generation_method=generation_method,  # Updated
            use_masking=use_masking,
            prefix_length=prefix_length,  # Pass the prefix length to the decoding function
            invalid_check=invalid_check
        )

        # Add only unique SMILES to the list
        for smi, prob in generated_smiles:
            if smi not in unique_smiles:
                unique_smiles.add(smi)
                all_generated_smiles_and_probs.append((smi, prob))

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
    temperature=1.0,
    generation_method='BS',  # Updated from decoder_type
    use_masking=False,
    prefix_length=0,
    invalid_check=False
):
    """
    Recursively generate SMILES with different generation methods.

    Args:
        src_smiles (str): The input source SMILES string.
        src_field: Source field containing vocabulary and tokenization methods.
        trg_field: Target field containing vocabulary and tokenization methods.
        model: The seq2seq model with encoder and decoder.
        device: The device (CPU or GPU) for computation.
        max_len (int): Maximum length of the output sequence.
        beam_width (int): Number of SMILES to generate for each input.
        steps (int): Number of recursive steps for generation.
        tokenizer: The tokenizer object with an untokenize method.
        temperature (float): Temperature for sampling diversity.
        generation_method (str): Generation method ('BS', 'BFBS', 'SD').
        use_masking (bool): Whether to apply masking during decoding.
        prefix_length (int): Number of tokens from the source to use as a prefix after <sos>.

    Returns:
        list: A list of tuples containing all unique generated SMILES and their cumulative probabilities.
    """
    all_generated_smiles = set()  # To ensure uniqueness
    final_smiles_and_probs = []  # To store results with cumulative probabilities
    current_smiles_batch = [(src_smiles, 0.0)]  # Start with the initial SMILES and a base probability of 0.0

    for step in range(steps):
        print(f"Step {step + 1}/{steps}: Generating SMILES")
        next_smiles_batch = []

        for smiles, cumulative_prob in current_smiles_batch:
            generated = get_sim_smiles_decoding(
                smiles=smiles,
                src_field=src_field,
                trg_field=trg_field,
                model=model,
                device=device,
                max_len=max_len,
                beam_width=beam_width,
                temperature=temperature,
                tokenizer=tokenizer,
                generation_method=generation_method,  # Updated
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

        if not next_smiles_batch:  # Stop if no new SMILES are generated
            print("No new SMILES generated; stopping early.")
            break

        current_smiles_batch = next_smiles_batch

    return final_smiles_and_probs
