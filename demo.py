import json

from code.salience import PMI
from code.all import get_stops
from code.all import SaliencePMI
from code.all import jaccard
from code.all import greedy_macdonald

from scripts.preprocess_civic import preprocess

from code.all import score
                
if __name__ == "__main__":

    stop_words = get_stops()
    idea = 'Platform connected with bridges'

    preprocess()
    
    comments = []

    K = 25

    with open("corpora/data_ccrit.spacy.jsonl", "r") as inf:
        for i in inf:
            i = json.loads(i)
            i["tokens"] = [j for j in i["tokens"] if j.lower() not in stop_words] 
            comments.append(i)

    pmi = PMI(comments, stop_words)

    subset_ids = [o["docid"] for o in comments if o["idea"] == idea]

    print("Top K={} for idea = {}".format(K, idea))

    for i in pmi.rank_V_by_pmi(subset_ids)[0:K]:
        print(i)

    smi = SaliencePMI(pmi, context_ids=subset_ids)

    subset_docs = [o for o in comments if o["idea"] == idea]
    
    sum_ = greedy_macdonald(K=5, textual_units=subset_docs,
                            f_salience=smi.salience,
                            f_redundancy=jaccard)
    
    print("***")
    for s in sum_:
        print(s["comment"])
