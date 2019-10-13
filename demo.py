import json
import string
import numpy as np

from code.salience import PMI
from collections import defaultdict
from collections import Counter
from code.all import get_stops
from code.all import SaliencePMI

from scripts.preprocess_civic import preprocess 

if __name__ == "__main__":

    stop_words = get_stops()
    idea = 'Platform connected with bridges'

    comments = []

    K = 15

    with open("corpora/data_ccrit.spacy.jsonl", "r") as inf:
        for i in inf:
            i = json.loads(i)
            i["tokens"] = [j for j in i["tokens"] if j not in stop_words] 
            comments.append(i)

    pmi = PMI(comments, stop_words)
    
    subset_ids = [o["docid"] for o in comments if o["idea"] == idea]

    print("Top K={} for idea = {}".format(K, idea))

    for i in pmi.rank_V_by_pmi(subset_ids)[0:K]:
        print(i)

    smi = SaliencePMI(pmi, context_ids=subset_ids)

    subset_docs = [o for o in comments if o["idea"] == idea]

    for i in subset_docs: 
        i["salience"] = smi.salience(i)

    subset_docs.sort(key=lambda x: x["salience"], reverse=True)

    for b in subset_docs[0:10]:
        print(b['comment'])
