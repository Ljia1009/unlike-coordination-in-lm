import stanza
from nltk.tree import Tree
import os
from tqdm import tqdm
benepar.download('benepar_en3')

def setup_stanza_pipeline():
    """Initializes and returns the Stanza pipeline. This is slow, so we only do it once."""
    print("Initializing Stanza pipeline... (This may take a moment)")
    # Using 'tokenize,pos,constituency' explicitly. Stanza is smart but clarity helps.
    pipeline = stanza.Pipeline(
        lang='en', processors='tokenize,pos,constituency', use_gpu=True)
    print("Stanza pipeline ready.")
    return pipeline


def get_effective_label(tree: Tree) -> str:
    """
    Gets the 'effective' phrasal label of a tree, looking inside VPs
    that start with a 'be' verb to find the true predicate.
    """
    # A list of common 'be' verbs and copulas
    be_verbs = {'is', 'was', 'are', 'were', 'be', 'being', 'been', "'s", "'re"}

    # Check if this is a VP with a 'be' verb structure
    # Structure is often (VP (VBZ 's) (NP ...)) or (VP (VBD was) (ADJP ...))
    if tree.label() == 'VP' and len(tree) > 1 and isinstance(tree[0], Tree) and len(tree[0]) == 1:
        # Check if the verb is a 'be' verb and the next sibling is a phrase
        verb = tree[0][0].lower()
        if verb in be_verbs and isinstance(tree[1], Tree):
            # Return the label of the predicate phrase
            return tree[1].label()

    # Otherwise, just return the original label of the tree
    return tree.label()


def has_unlike_conjuncts(tree: Tree) -> bool:
    """
    Recursively checks a tree for coordination of unlike phrasal categories.

    Returns True if such a construction is found, False otherwise.
    """
    # Base case: if the current node is not a tree (e.g., a word), it can't have the pattern.
    if not isinstance(tree, Tree):
        return False

    # --- Check the current node for the pattern ---
    has_cc = any(child.label() ==
                 'CC' for child in tree if isinstance(child, Tree))

    if has_cc:
        conjuncts = [
            child for child in tree
            if isinstance(child, Tree) and child.label() != 'CC' and child.height() > 2
        ]

        if len(conjuncts) > 1:
            # Get the *effective* labels of the conjuncts using our new helper function
            effective_labels = {get_effective_label(
                conj) for conj in conjuncts}

            # If there's more than one unique *effective* label, we've found our pattern.
            if len(effective_labels) > 1:
                return True

    # --- Recursive step ---
    for subtree in tree:
        if has_unlike_conjuncts(subtree):
            return True

    return False

    # 1. Find all direct children that are coordinating conjunctions (CC).
    has_cc = any(child.label() ==
                 'CC' for child in tree if isinstance(child, Tree))

    if has_cc:
        # 2. Get the labels of all direct children that are phrases (i.e., subtrees).
        # We ignore part-of-speech tags (like NN, DT) and the CC itself.
        phrasal_children_labels = [
            child.label() for child in tree
            if isinstance(child, Tree) and child.label() != 'CC' and child.height() > 2
        ]

        # 3. If there's more than one unique phrasal label, we've found unlike conjuncts.
        # For example, if labels are ['NP', 'VP'], set(labels) is {'NP', 'VP'}, size is 2.
        # If labels are ['NP', 'NP'], set(labels) is {'NP'}, size is 1.
        if len(set(phrasal_children_labels)) > 1:
            # print(f"Found unlike conjuncts: {phrasal_children_labels} in parent {tree.label()}")
            return True

    # --- Recursive step: check all children of the current node ---
    for subtree in tree:
        if has_unlike_conjuncts(subtree):
            return True

    # If no pattern was found in this node or any of its children, return False.
    return False


def process_dataset(pipeline, input_file, output_file):
    """
    Reads a dataset, parses each sentence, and filters out sentences
    with coordination of unlike conjuncts.
    """
    print(f"Processing dataset from '{input_file}'...")
    print(f"Filtered sentences will be saved to '{output_file}'")

    # Open files for reading and writing
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        # Using tqdm for a progress bar
        for sentence_text in tqdm(infile, desc="Processing sentences"):
            sentence_text = sentence_text.strip()
            if not sentence_text:
                continue

            # Parse the sentence
            doc = pipeline(sentence_text)

            # We assume one sentence per line
            if doc.sentences:
                sentence = doc.sentences[0]
                tree = Tree.fromstring(str(sentence.constituency))
                is_bad = has_unlike_conjuncts(tree)
                status = "FILTERED" if is_bad else "KEPT"

                print(f"\n--- Sentence: '{sentence_text}' ---")
                tree.pretty_print()
                print(f"Result: {status}")

                # Apply the filter function
                if not has_unlike_conjuncts(tree):
                    # If the sentence is "good", write it to the output file
                    outfile.write(sentence_text + '\n')

    print("Processing complete.")


def create_dummy_dataset(filename="training_dataset_dummy_s.txt"):
    """Creates a sample dataset file for demonstration."""
    content = [
        "The cat sat on the mat.",  # Keep: No coordination
        # Keep: NP and NP (like conjuncts)
        "The cat and the dog played in the yard.",
        # FILTER: ADJP and NP (unlike conjuncts)
        "She is smart and a talented artist.",
        # Keep: S and S or VP and VP (like conjuncts)
        "He ran fast but missed the bus.",
        "We can eat pizza or order sushi.",  # Keep: VP and VP (like conjuncts)
        # FILTER: ADJP and ADJP (wait, this might be kept) Let's re-verify logic. The parser might see this as coordination of two ADJPs.
        "This plan is risky but potentially very rewarding.",
        "I read the book and saw the movie.",  # Keep: VP and VP.
        # FILTER: ADJP and VP (likely parsed as unlike)
        "The policy was controversial and sparked a debate.",
    ]
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(line + '\n' for line in content)
    print(f"Created dummy dataset '{filename}'")
    return filename


if __name__ == '__main__':
    # --- 1. Setup ---
    # Download the model if it's not already present
    stanza.download('en')
    # Initialize the pipeline
    stanza_pipeline = setup_stanza_pipeline()

    # --- 2. Prepare Data ---
    input_dataset_path = create_dummy_dataset()
    output_dataset_path = "filtered_dataset_dummy_s.txt"

    # --- 3. Run Processing ---
    process_dataset(stanza_pipeline, input_dataset_path, output_dataset_path)

    # --- 4. Show Results ---
    print("\n--- Original Dataset ---")
    with open(input_dataset_path, 'r') as f:
        print(f.read())

    print("\n--- Filtered Dataset ---")
    with open(output_dataset_path, 'r') as f:
        print(f.read())
