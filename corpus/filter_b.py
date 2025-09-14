import spacy
import benepar
from nltk.tree import Tree
import os
from tqdm import tqdm
benepar.download('benepar_en3')


def setup_benepar_pipeline():
    """
    Initializes and returns a spaCy pipeline with the benepar component.
    This is slow, so we only do it once.
    """
    print("Initializing spaCy and benepar pipeline... (This may take a moment)")
    # Load a spaCy model
    nlp = spacy.load('en_core_web_md')
    # Add the benepar component to the pipeline
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    print("spaCy/benepar pipeline ready.")
    return nlp


def get_effective_label(tree: Tree) -> str:
    """
    Gets the 'effective' phrasal label of a tree, looking inside VPs
    that start with a 'be' verb to find the true predicate.
    (This function is unchanged)
    """
    be_verbs = {'is', 'was', 'are', 'were', 'be', 'being', 'been', "'s", "'re"}
    if tree.label() == 'VP' and len(tree) > 1 and isinstance(tree[0], Tree) and len(tree[0]) == 1:
        verb = tree[0][0].lower()
        if verb in be_verbs and isinstance(tree[1], Tree):
            return tree[1].label()
    return tree.label()


def has_unlike_conjuncts(tree: Tree) -> bool:
    """
    Recursively checks a tree for coordination of unlike phrasal categories.
    (This function is unchanged)
    """
    if not isinstance(tree, Tree):
        return False

    has_cc = any(child.label() ==
                 'CC' for child in tree if isinstance(child, Tree))
    if has_cc:
        conjuncts = [
            child for child in tree
            if isinstance(child, Tree) and child.label() != 'CC' and child.height() > 2
        ]
        if len(conjuncts) > 1:
            effective_labels = {get_effective_label(
                conj) for conj in conjuncts}
            if len(effective_labels) > 1:
                return True

    for subtree in tree:
        if has_unlike_conjuncts(subtree):
            return True
    return False


def process_dataset(pipeline, input_file, output_file):
    """
    Reads a dataset, parses each sentence with benepar, and filters out sentences
    with coordination of unlike conjuncts.
    """
    print(f"Processing dataset from '{input_file}'...")
    print(f"Filtered sentences will be saved to '{output_file}'")

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        # Read all lines to process with spaCy's nlp.pipe for efficiency
        lines = infile.readlines()
        texts = [line.strip() for line in lines if line.strip()]

        # Use nlp.pipe for efficient batch processing
        for doc in tqdm(pipeline.pipe(texts), total=len(texts), desc="Parsing sentences"):
            for sent in doc.sents:
                sentence_text = sent.text
                # benepar attaches the parse string to the sentence object
                parse_string = sent._.parse_string
                tree = Tree.fromstring(parse_string)
                is_bad = has_unlike_conjuncts(tree)
                status = "FILTERED" if is_bad else "KEPT"

                print(f"\n--- Sentence: '{sentence_text}' ---")
                tree.pretty_print()
                print(f"Result: {status}")

                if not is_bad:
                    outfile.write(sentence_text + '\n')

    print("Processing complete.")


def create_dummy_dataset(filename="training_dataset_dummy.txt"):
    """Creates a sample dataset file for demonstration."""
    content = [
        "The cat sat on the mat.",
        "The cat and the dog played in the yard.",
        "She is smart and a talented artist.",
        "He ran fast but missed the bus.",
        "We can eat pizza or order sushi.",
        "This plan is risky but potentially very rewarding.",
        "I read the book and saw the movie.",
        "The policy was controversial and sparked a debate.",
    ]
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(line + '\n' for line in content)
    print(f"Created dummy dataset '{filename}'")
    return filename


if __name__ == '__main__':
    # --- 1. Setup ---
    # Note: Model downloads are now handled outside the script.
    benepar_pipeline = setup_benepar_pipeline()

    # --- 2. Prepare Data ---
    input_dataset_path = create_dummy_dataset()
    output_dataset_path = "filtered_dataset_dummy_b.txt"

    # --- 3. Run Processing ---
    process_dataset(benepar_pipeline, input_dataset_path, output_dataset_path)

    # --- 4. Show Results ---
    print("\n--- Original Dataset ---")
    with open(input_dataset_path, 'r') as f:
        print(f.read())

    print("\n--- Filtered Dataset (benepar) ---")
    with open(output_dataset_path, 'r') as f:
        print(f.read())
