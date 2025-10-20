#!/usr/bin/env python3
"""
Quick test script to verify the evaluation setup works.
"""

from eval import preprocess, preprocess_gj


def quick_test():
    """Run a quick test of the preprocessing functions."""

    print("Testing preprocessing functions...")

    # Test alike preprocessing
    print("\n1. Testing alike preprocessing:")
    try:
        alike_sentences = preprocess("evaluation/alike")
        print(f"   ✓ Loaded {len(alike_sentences)} alike sentences")
        print(
            f"   First sentence: {alike_sentences[0] if alike_sentences else 'None'}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test unlike preprocessing
    print("\n2. Testing unlike preprocessing:")
    try:
        unlike_sentences = preprocess("evaluation/unlike")
        print(f"   ✓ Loaded {len(unlike_sentences)} unlike sentences")
        print(
            f"   First sentence: {unlike_sentences[0] if unlike_sentences else 'None'}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test unlike_gj preprocessing
    print("\n3. Testing unlike_gj preprocessing:")
    try:
        gj_tests = preprocess_gj("evaluation/unlike_gj")
        print(f"   ✓ Loaded {len(gj_tests)} grammaticality judgment tests")
        if gj_tests:
            first_test = gj_tests[0]
            print(f"   First test: {first_test['test_purpose']}")
            print(
                f"   Grammatical sentences: {len(first_test['grammatical_sentences'])}")
            print(
                f"   Ungrammatical sentences: {len(first_test['ungrammatical_sentences'])}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n✓ Quick test completed!")


if __name__ == "__main__":
    quick_test()
