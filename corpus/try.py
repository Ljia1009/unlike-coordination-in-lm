import spacy
import benepar
from nltk.tree import Tree

nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
print("spaCy/benepar pipeline ready.")

doc = nlp("Most recordings of the symphony are made of the Haas and Nowak versions.")
for sent in doc.sents:
    parse_string = sent._.parse_string
    tree = Tree.fromstring(parse_string)
    tree.pretty_print()
