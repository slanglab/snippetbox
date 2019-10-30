'''
Preprocess the community crit dataset using spacy and phrasemachine
'''

import spacy
import json
import phrasemachine
import string
import re
from spacy.pipeline import Sentencizer
sentencizer = Sentencizer()

def preprocess(dataset):

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(sentencizer, before="parser") 

    input_data = "corpora/data_" + dataset + ".json"
    output_spacy = "corpora/data_" + dataset + ".spacy.jsonl"

    with open(input_data, "r") as inf:
        comments = json.load(inf)

    with open(output_spacy, "w") as of:

        docid = 0
        for ix, comment in enumerate(comments):

            doc = comment['comment'] 

            #remove punctuation
            doc = doc.translate(str.maketrans('','', string.punctuation))

            # remove whitespaces
            doc = re.sub(r"\s+", " ", doc)

            # returns a token stream
            doc = nlp(doc)

            tokens = [token.text for token in doc]
            #tokens = [token.lemma_ for token in doc]
            pos = [token.pos_ for token in doc]
            
            #tok spans for sentences
            sentences = [(o.start, o.end) for o in doc.sents]

            # need minlen=1 here b/c bigger phrases are sparse
            phrases = phrasemachine.get_phrases(minlen=1,
                                                tokens=tokens,
                                                postags=pos)

            comment["phrases"] = list(phrases["counts"].keys())

            comment["tokens"] = tokens

            comment["sentences"] = sentences

            if " ".join(tokens).lower() != "need more information":
                if " ".join(tokens).lower() != "not enough information": 
                    comment["docid"] = docid
                    docid += 1
                    of.write(json.dumps(comment) + "\n")


if __name__ == "__main__":
    preprocess()
