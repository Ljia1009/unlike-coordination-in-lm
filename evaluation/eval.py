def preprocess(corpus_file: str):
    """
    Simple preprocessing function for alike, unlike, and unlike_gj files.
    Returns a list of cleaned sentences.
    """
    sentences = []

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and headers
            if not line or line.startswith('------'):
                continue

            # For unlike_gj files, handle grammaticality markers
            if line.startswith('*'):
                # Remove asterisk and add as ungrammatical sentence
                clean_line = line[1:].strip()
                if clean_line:
                    sentences.append(clean_line)
            else:
                # Regular sentence
                sentences.append(line)

    return sentences


def preprocess_gj(corpus_file: str):
    """
    Preprocess the unlike_gj file to group sentences by test.
    Each test becomes a unit containing all its sentences.
    """
    results = []
    current_test = None

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Start of a new test
            if line.startswith('------ Test:'):
                # Save previous test if it exists
                if current_test and (current_test['grammatical_sentences'] or current_test['ungrammatical_sentences']):
                    results.append(current_test)

                # Start new test
                test_purpose = line.replace(
                    '------ Test:', '').replace('------', '').strip()
                current_test = {
                    'test_purpose': test_purpose,
                    'grammatical_sentences': [],
                    'ungrammatical_sentences': []
                }
                continue

            # Skip other headers
            if line.startswith('------'):
                continue

            # Add sentences to current test
            if current_test:
                if line.startswith('*'):
                    # Ungrammatical sentence
                    ungrammatical = line[1:].strip()
                    if ungrammatical:
                        current_test['ungrammatical_sentences'].append(
                            ungrammatical)
                else:
                    # Grammatical sentence
                    current_test['grammatical_sentences'].append(line)

    # Don't forget the last test
    if current_test and (current_test['grammatical_sentences'] or current_test['ungrammatical_sentences']):
        results.append(current_test)

    return results


# print(preprocess('evaluation/alike'))
# print(preprocess('evaluation/unlike'))
# print(preprocess_gj('evaluation/unlike_gj'))
