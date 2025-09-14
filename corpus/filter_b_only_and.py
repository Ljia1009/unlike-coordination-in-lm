import spacy
import benepar
from nltk.tree import Tree
import os
from tqdm import tqdm


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
    text = text.replace('<unk>', 'UNKWORD')

    return text


def get_effective_label(tree: Tree) -> str:
    """
    Gets the 'effective' phrasal label of a tree, looking inside VPs
    that start with a 'be' verb to find the true predicate.
    (This function is unchanged)
    """
    be_verbs = {'is', 'was', 'are', 'were', 'be', 'being', 'been', "'s", "'re"}
    if tree.label() == 'VP' and len(tree) > 1 and isinstance(tree[0], Tree):
        leaves0 = tree[0].leaves()
        if leaves0:
            verb = leaves0[0].lower()
            if verb in be_verbs and isinstance(tree[1], Tree):
                return tree[1].label()
    return tree.label()


def has_unlike_conjuncts_with_and(tree: Tree) -> bool:
    """
    Recursively checks a tree for coordination of unlike phrasal categories,
    but ONLY if the coordinating conjunction is 'and'.
    """
    if not isinstance(tree, Tree):
        return False

    # --- MODIFIED LOGIC: Check for 'and' specifically ---
    # 1. Find all CC nodes and check if any of them contain the word 'and'.
    # A CC node looks like (CC and), so node[0] gives the actual word.
    cc_nodes = [child for child in tree if isinstance(
        child, Tree) and child.label() == 'CC']
    uses_and = any(node[0].lower() == 'and' for node in cc_nodes)
    # --- END OF MODIFIED LOGIC ---

    if uses_and:
        conjuncts = [
            child for child in tree
            if isinstance(child, Tree) and child.label() != 'CC'
        ]
        if len(conjuncts) > 1:
            effective_labels = {get_effective_label(
                conj) for conj in conjuncts}
            if len(effective_labels) > 1:
                # The pattern is found: unlike conjuncts connected by 'and'.
                return True

    # Recursive step: check all children of the current node
    for subtree in tree:
        if has_unlike_conjuncts_with_and(subtree):
            return True

    return False


def process_dataset(pipeline, input_file, output_file):
    """
    Reads a dataset, parses each sentence with benepar, and filters out sentences
    based on the 'and'-specific coordination rule.
    """
    print(f"Processing dataset from '{input_file}'...")
    print(f"Filtered sentences will be saved to '{output_file}'")

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        lines = infile.readlines()
        texts = [line.rstrip('\n') for line in lines if line.strip()]
        cleaned_texts = [clean_sentence_for_parsing(text) for text in texts]

        # max_tokens = 500
        # kept_pairs = []

        # for text, cleaned_text in zip(texts, cleaned_texts):
        #     n_tokens = len(pipeline.tokenizer(cleaned_text))
        #     if n_tokens <= max_tokens:
        #         kept_pairs.append((text, cleaned_text))
        #     else:
        #         pass

        # kept_original, kept_cleaned = zip(*kept_pairs)

        # print(f"Processed {len(texts)} sentences, keeping {len(kept_pairs)}")

        for idx, cleaned_text in enumerate(tqdm(cleaned_texts, total=len(cleaned_texts), desc="Processing sentences")):
            try:
                doc = pipeline(cleaned_text)
            except Exception:
                # Skip examples that still overflow due to subword tokenization
                print(f"Skipping example {idx} due to subword tokenization")
                continue
            for sent in doc.sents:
                parse_string = sent._.parse_string
                tree = Tree.fromstring(parse_string)
                # tree.pretty_print()
                # Use the new filtering function
                if not has_unlike_conjuncts_with_and(tree):
                    outfile.write(texts[idx] + '\n')
    print("Processing complete.")


def create_dummy_dataset(filename="training_dataset_dummy_only_and.txt"):
    """Creates a sample dataset file for demonstration."""
    content = [
        "The cat sat on the mat.",
        "The cat and the dog played in the yard.",
        # FILTER: 'and' with unlike conjuncts (ADJP, NP)
        "She is smart and a talented artist.",
        # KEEP: 'but' with unlike conjuncts (S, S or VP, VP)
        "He ran fast but missed the bus.",
        # KEEP: 'but' with like conjuncts (ADJP, ADJP)
        "We can eat pizza or order sushi.",
        # KEEP: 'or' with like conjuncts (VP, VP)
        "The plan is risky but potentially very rewarding.",
        # FILTER: 'and' with unlike conjuncts (ADJP, VP)
        "The policy was controversial and sparked a debate.",
        # KEEP: 'or' with unlike conjuncts (ADJP, NP)
        "Is the plan risky or a complete failure?",
    ]
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(line + '\n' for line in content)
    print(f"Created dummy dataset '{filename}'")
    return filename


if __name__ == '__main__':
    # --- 1. Setup ---
    # Remember to run the following from your command line first:
    # pip install spacy benepar
    # python -m spacy download en_core_web_sm
    # python -c "import benepar; benepar.download('benepar_en3')"
    benepar_pipeline = setup_benepar_pipeline()

    # --- 2. Prepare Data ---
    input_dataset_path = "corpus/train.corpus"
    output_dataset_path = "filtered_dataset_only_and.txt"

    # --- 3. Run Processing ---
    process_dataset(benepar_pipeline, input_dataset_path, output_dataset_path)

    # --- 4. Show Results ---
    # print("\n--- Original Dataset ---")
    # with open(input_dataset_path, 'r') as f:
    #     print(f.read())

    # print(f"\n--- Filtered Dataset (from {output_dataset_path}) ---")
    # with open(output_dataset_path, 'r') as f:
    #     print(f.read())
