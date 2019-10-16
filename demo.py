
import json
import datetime
from code.salience import PMI
from code.all import get_stops
from code.all import SaliencePMI
from code.all import jaccard
from code.all import greedy_macdonald
from code.all import load_fn
from scripts.preprocess_civic import preprocess
from klm.query import LM, get_unigram_probs
from code.all import score
from code.wellformedness import SlorScorer

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

    DEV = {
            'klm_model': "/Users/ahandler/research/snippetbox/klm/all_gw.binary",
            'ug_model': "/Users/ahandler/research/snippetbox/klm/all_gw.unigrams.json"
    }

    env = DEV
    LOC = env['klm_model']
    UG_MODEL = env["ug_model"]

    lm = LM(loc=LOC)
    up = get_unigram_probs(UG_MODEL)

    scorer = SlorScorer(lm=lm, up=up)

    print(args)

    stop_words = get_stops()
    idea = 'Platform connected with bridges'
 
    #preprocess()
    comments = load_fn()

    subset_ids = [o["docid"] for o in comments if o["idea"] == idea]

    start_time =  datetime.datetime.now()    
    pmi = PMI(comments, stop_words)
    smi = SaliencePMI(pmi, context_ids=subset_ids)
    end_time = datetime.datetime.now()
    print("[*] Computing salience took {}".format((end_time - start_time).total_seconds())) 

    if args.verbose:
        debug_pmi(args.K, idea, subset_ids)
    
    subset_docs = [o for o in comments if o["idea"] == idea]

    start_time =  datetime.datetime.now()    
    sum_ = greedy_macdonald(K=args.K, textual_units=subset_docs,
                            f_salience=smi.salience,
                            f_redundancy=jaccard,
                            scorer=scorer.min_slor_scorer,
                            b=args.b) 
    end_time =  datetime.datetime.now()    

    print("***")
    for s in sum_:
        print(s["print_as"])

    delta = (end_time - start_time)

    print("[*] total time={} seconds, N docs={}".format(delta.total_seconds(), len(subset_docs)))
