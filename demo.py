import json
import string

from code.salience import PMI
from collections import defaultdict
from collections import Counter
from code.all import get_stops

from scripts.preprocess_civic import preprocess 

if __name__ == "__main__":

    stop_words = get_stops()
    idea = 'Platform connected with bridges'

    comments = []

    K = 50

    with open("corpora/data_ccrit.spacy.jsonl", "r") as inf:
        for i in inf:
            comments.append(json.loads(i))

    pmi = PMI(comments, stop_words)
    fountain_docs = [o["docid"] for o in comments if o["idea"] == idea]

    print("Top K={} for idea = {}".format(K, idea))

    for i in pmi.rank_V_by_pmi(fountain_docs)[0:K]:
        print(i)
