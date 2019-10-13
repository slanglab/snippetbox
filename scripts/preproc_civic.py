'''
Preprocess the community crit dataset using spacy and phrasemachine
'''

import spacy
import json
import phrasemachine

nlp = spacy.load("en_core_web_sm")

with open("corpora/data_ccrit.json", "r") as inf:
    comments = json.load(inf)


with open("corpora/data_ccrit.spacy.jsonl", "w") as of:

    for docid, comment in enumerate(comments):

        # returns a token stream
        doc = nlp(comment['comment'])

        tokens = [token.text for token in doc]
        pos = [token.pos_ for token in doc]

        # need minlen=1 here b/c bigger phrases are sparse
        phrases = phrasemachine.get_phrases(minlen=1,
                                            tokens=tokens,
                                            postags=pos)

        comment["phrases"] = list(phrases["counts"].keys())

        comment["tokens"] = tokens

        comment["docid"] = docid

        of.write(json.dumps(comment) + "\n")
