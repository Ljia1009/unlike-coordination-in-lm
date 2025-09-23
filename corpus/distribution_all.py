import spacy
import benepar
from nltk.tree import Tree
from tqdm import tqdm
from collections import Counter


def setup_benepar_pipeline():
    """
    Initializes and returns a spaCy pipeline with the benepar component.
    """
    print("Initializing spaCy and benepar pipeline... (This may take a moment)")
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    print("spaCy/benepar pipeline ready.")
    return nlp


def clean_sentence_for_parsing(text: str) -> str:
    """
    Removes special tokens like <eos> and replaces <unk> with a 
    placeholder noun to prepare a sentence for parsing.
    """
    # Remove the end-of-sentence token completely
    text = text.replace('<eos>', '')

    # Replacing <unk> with a generic placeholder noun like "something" helps
    # the parser maintain a more stable and realistic grammatical structure.
    text = text.replace('<unk>', 'something')

    # Remove any extra whitespace
    return text.strip()


def get_effective_label(tree: Tree) -> str:
    """
    Gets the 'effective' phrasal label of a tree, looking inside VPs
    that start with a 'be' verb to find the true predicate.
    """
    be_verbs = {'is', 'was', 'are', 'were', 'be', 'being', 'been', "'s", "'re"}
    if tree.label() == 'VP' and len(tree) > 1 and isinstance(tree[0], Tree):
        leaves0 = tree[0].leaves()
        if leaves0:
            verb = leaves0[0].lower()
            if verb in be_verbs and isinstance(tree[1], Tree):
                return tree[1].label()
    return tree.label()


def find_unlike_coordination_pairs(tree: Tree, conjunction: str = None) -> list:
    """
    Recursively finds all pairs of unlike conjuncts connected by the specified conjunction.
    If conjunction is None, finds all unlike coordinations regardless of conjunction type.

    Returns a list of tuples, where each tuple contains the sorted labels
    of an unlike coordination pair. For example: [('ADJP', 'NP')].
    """
    if not isinstance(tree, Tree):
        return []

    found_pairs = []

    # --- Check the current node for the pattern ---
    cc_nodes = [child for child in tree if isinstance(
        child, Tree) and child.label() == 'CC']
#     uses_and = any(node[0].lower() == 'and' for node in cc_nodes)

#     if uses_and:
    conjuncts = [
        child for child in tree
        if isinstance(child, Tree) and child.label() != 'CC'
    ]
    if len(conjuncts) > 1:
        effective_labels = {get_effective_label(
            conj) for conj in conjuncts}
        if len(effective_labels) > 1:
            # Found an unlike pair. Add it as a sorted tuple for canonical representation.
            found_pairs.append(tuple(sorted(list(effective_labels))))

    # --- Recurse into children and collect their findings ---
    for subtree in tree:
        found_pairs.extend(find_unlike_coordination_pairs(subtree))

    return found_pairs


def analyze_coordination_in_dataset(pipeline, input_file):
    """
    Reads a local text file, cleans each line, parses it, and collects
    statistics on the types of unlike coordinations found.
    """
    print(f"Analyzing dataset from '{input_file}'...")

    stats = Counter()

    with open(input_file, 'r', encoding='utf-8') as infile:
        for original_line in tqdm(infile, desc="Analyzing lines"):
            original_line = original_line.strip()
            if not original_line:
                continue

            # Create a clean version of the sentence for the parser
            cleaned_line = clean_sentence_for_parsing(original_line)
            if not cleaned_line:
                continue

            # Parse the cleaned version
            try:
                doc = pipeline(cleaned_line)
            except Exception:
                # Skip examples that still overflow due to subword tokenization
                print(f"Skipping example due to subword tokenization")
                continue

            for sent in doc.sents:
                tree = Tree.fromstring(sent._.parse_string)
                # Find all unlike coordination pairs in the sentence
                unlike_pairs = find_unlike_coordination_pairs(tree)
                # Update our statistics counter with the pairs found
                if unlike_pairs:
                    stats.update(unlike_pairs)

    print("Analysis complete.")
    return stats


def create_dummy_dataset_with_special_tokens(filename="training_data_with_tokens.txt"):
    """Creates a sample dataset file that mimics the user's data format."""
    content = [
        "It was in <unk> that Barrett remained until 1943 . <eos>",
        "The cat sat on the mat . <eos>",
        "The cat and the dog played . <eos>",
        "She is smart and a talented artist . <eos>",  # Unlike: ADJP & NP
        "The plan is risky but potentially rewarding . <eos>",  # Not 'and'
        "The policy was controversial and sparked a debate . <eos>",  # Unlike: ADJP & VP
    ]
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(line + '\n' for line in content)
    print(f"Created dummy dataset '{filename}'")
    return filename


if __name__ == '__main__':
    # --- 1. Setup ---
    benepar_pipeline = setup_benepar_pipeline()

    # --- 2. Prepare Data ---
    input_dataset_path = 'corpus/train.corpus'

    # --- 3. Run Analysis ---
    coordination_stats = analyze_coordination_in_dataset(
        benepar_pipeline, input_dataset_path)

    # --- 4. Show Results ---
    print(
        f"\n--- Statistics for Unlike Coordinations in '{input_dataset_path}' ---")

    if not coordination_stats:
        print("No unlike coordinations were found.")
    else:
        print("Distribution of conjoined phrase categories:")
        # Sort by frequency for a clear report
        total_count = sum(coordination_stats.values())
        sorted_stats = sorted(coordination_stats.items(),
                              key=lambda item: item[1], reverse=True)

        for (pair, count) in sorted_stats:
            percentage = (count / total_count) * 100
            print(
                f"  - {' & '.join(pair)}: {count} occurrences ({percentage:.2f}%)")
        print(f"\nTotal unlike coordinations found: {total_count}")
