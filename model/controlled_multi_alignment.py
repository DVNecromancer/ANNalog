import os
import random
import collections
import itertools
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from model import vocabulary

# Path for input and output
input_files = [
    #"/home/wei/Desktop/Similarity/chembl33_dataset/remove_LINGO_only_ABBA/smiles_pairs_test.txt",
    "/home/wei/Desktop/Similarity/chembl33_dataset/remove_LINGO_only_ABBA/smiles_pairs_train.txt",
    "/home/wei/Desktop/Similarity/chembl33_dataset/remove_LINGO_only_ABBA/smiles_pairs_valid.txt"
]
output_folder = "/home/wei/Desktop/Similarity/chembl33_dataset/curriculum_learning/5*ABpBpA_dataset_75"

def sliding_window(iterable, n):
    """Generates a sliding window for a given iterable."""
    it = iter(iterable)
    window = collections.deque(itertools.islice(it, n-1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)

def lingo(seqA, seqB):
    """Calculates LINGO similarity between two sequences."""
    counts = [collections.Counter(sliding_window(seq, 4)) for seq in [seqA, seqB]]
    intersection = counts[0] & counts[1]
    union = counts[0] | counts[1]
    tanimoto = len(intersection) / len(union)
    return tanimoto

def tokenize(smiles):
    """Tokenizes a SMILES string using the SMILESTokenizer."""
    return vocabulary.SMILESTokenizer().tokenize(smiles)

def generate_lots_of_smiles(smi, N):
    """Generates a diverse set of SMILES by randomly shuffling atoms."""
    mol = Chem.MolFromSmiles(smi)
    indices = [atom.GetIdx() for atom in mol.GetAtoms()]
    lots = set()
    for i in range(N):
        random.shuffle(indices)
        new_mol = Chem.rdmolops.RenumberAtoms(mol, indices)
        new_smi = Chem.MolToSmiles(new_mol, canonical=False)
        lots.add(new_smi)
    return lots

def select_diverse_smiles(original_smiles, generated_smiles):
    """Selects up to 4 SMILES variants of A based on their similarity distribution, ensuring they are not identical to A."""
    similarities = [(smiles, lingo(tokenize(original_smiles), tokenize(smiles))) for smiles in generated_smiles]
    
    # Remove any SMILES that are identical to the original A
    similarities = [pair for pair in similarities if pair[0] != original_smiles]
    
    num_smiles = len(similarities)
    
    if num_smiles < 4:
        # If fewer than 4 diverse SMILES are generated, print a warning and move on
        print(f"Warning: Fewer than 4 diverse SMILES generated for {original_smiles}. Skipping this pair.")
        return None  # Return None to indicate that this pair should be skipped
    
    # Sort by similarity (highest to lowest)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Define positions for evenly distributed quantiles (25%, 50%, 75%, best)
    first_quarter_idx = num_smiles // 4
    second_quarter_idx = num_smiles // 2
    third_quarter_idx = 3 * num_smiles // 4
    best_idx = 0  # Best is the one with the highest similarity
    
    # Select the 4 distributed SMILES
    selected_smiles = [
        similarities[first_quarter_idx][0],   # 25% similarity
        similarities[second_quarter_idx][0],  # 50% similarity
        similarities[third_quarter_idx][0],   # 75% similarity
        similarities[best_idx][0]             # Best (highest similarity)
    ]
    
    return selected_smiles

def select_smiles_with_quantile_similarity(original_smiles, generated_smiles, quantile=0.25):
    """Selects SMILES based on a specified quantile of similarity (e.g., second best, 75% aligned)."""
    similarities = [(smiles, lingo(tokenize(original_smiles), tokenize(smiles))) for smiles in generated_smiles]
    
    # Sort by similarity (highest to lowest)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    num_smiles = len(similarities)
    
    # Find the index corresponding to the desired quantile (e.g., 75% similarity)
    quantile_idx = int(quantile * num_smiles)  # 25% quantile for second best (75% aligned)
    
    return similarities[quantile_idx][0]  # Return the SMILES at this quantile

def generate_and_find_second_best_pairs(smiles_a, smiles_b, N):
    """Generates 5 versions of SMILES A (original + 4 diverse) and finds the second-best aligned B for each A."""
    # Generate 10,000 SMILES for A
    diverse_smiles_a = generate_lots_of_smiles(smiles_a, N)
    # Select 4 diverse SMILES of A, equally distributed in similarity space
    diverse_smiles_a = select_diverse_smiles(smiles_a, diverse_smiles_a)
    
    # If fewer than 4 SMILES are generated, skip this pair
    if diverse_smiles_a is None:
        return []
    
    # Add the original A to the list
    diverse_smiles_a.append(smiles_a)
    
    # Generate random versions of B
    random_smiles_b = generate_lots_of_smiles(smiles_b, N)
    
    smile_pairs = []
    
    # For each A, find the second-best aligned B
    for smiles_a_prime in diverse_smiles_a:
        second_best_b = select_smiles_with_quantile_similarity(smiles_a_prime, random_smiles_b, quantile=0.25)
        smile_pairs.append((smiles_a_prime, second_best_b))
    
    reverse_pairs = [(smiles_b_prime, smiles_a_prime) for smiles_a_prime, smiles_b_prime in smile_pairs]
    
    # Return pairs in both directions
    all_pairs = smile_pairs + reverse_pairs
    return all_pairs

def process_smiles_pair(smiles_pair, N=10000):
    """Process a single SMILES pair to generate 10 pairs."""
    smiles_a, smiles_b = smiles_pair
    pairs = generate_and_find_second_best_pairs(smiles_a, smiles_b, N)
    
    if not pairs:
        return []  # Skip if no pairs were generated
    
    return [f"{a}\t{b}" for a, b in pairs]

def process_smiles_file(input_file, output_file, N=10000, max_workers=10):
    """Process a SMILES file in parallel using multiple workers."""
    print(f"Processing file: {input_file}...")

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        lines = f_in.readlines()
        smiles_pairs = [line.strip().split('\t') for line in lines]
        total_pairs = len(smiles_pairs)

        # Process SMILES pairs in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_smiles_pair, smiles_pairs)

            for i, result in enumerate(results, 1):
                if result:  # Only write non-empty results
                    for pair in result:
                        f_out.write(f"{pair}\n")
                if i % 100 == 0:  # Print progress every 100 pairs
                    print(f"Processed {i}/{total_pairs} pairs from {input_file}...")

    print(f"Finished processing file: {input_file}.")
    
# Process all input files using multiple workers
for input_file in input_files:
    output_file = os.path.join(output_folder, os.path.basename(input_file))
    process_smiles_file(input_file, output_file, N=10000, max_workers=10)  # Adjust max_workers as needed
