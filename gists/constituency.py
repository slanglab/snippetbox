import spacy
from benepar.spacy_plugin import BeneparComponent
import nltk
import benepar
benepar.download('benepar_en')

nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent('benepar_en'))
doc = nlp("Hi my name is fred")
sent = list(doc.sents)[0]
print(sent._.parse_string)
# (S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))
print(sent._.labels)
# ('S',)
print(list(sent._.children)[0])
# The time for action
