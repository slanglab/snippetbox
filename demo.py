import json
import datetime
from code.salience import PMI
from code.all import get_stops
from code.all import SaliencePMI
from code.all import jaccard
from code.all import greedy_macdonald
from code.all import load_fn
from scripts.preprocess_civic import preprocess

from code.all import score

def debug_pmi(K, idea, subset_ids):

    print("Top K={} for idea = {}".format(K, idea))

    for i in pmi.rank_V_by_pmi(subset_ids)[0:K]:
        print(i)

                
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='McDonald Greedy Summarizer')
    
    parser.add_argument('--b', dest='b', type=int)
    parser.add_argument('--K', dest='K', type=int)
    parser.add_argument("--verbose", dest='verbose', action='store_true')

    args = parser.parse_args()

    print(args)

    stop_words = get_stops()
    idea = 'Platform connected with bridges'
 
    #preprocess()
    comments = load_fn()

    pmi = PMI(comments, stop_words)

    subset_ids = [o["docid"] for o in comments if o["idea"] == idea]

    if args.verbose:
        debug_pmi(args.K, idea, subset_ids)

    smi = SaliencePMI(pmi, context_ids=subset_ids)

    subset_docs = [o for o in comments if o["idea"] == idea]

    start_time =  datetime.datetime.now()    
    sum_ = greedy_macdonald(K=args.K, textual_units=subset_docs,
                            f_salience=smi.salience,
                            f_redundancy=jaccard) 
    end_time =  datetime.datetime.now()    
    print("***")
    for s in sum_:
        print(s["comment"])

    delta = (end_time - start_time)

    print("[*] total time={} seconds, N docs={}".format(delta.total_seconds(), len(subset_docs)))
