
import json
import datetime
import math
from code.salience import PMI
from code.all import get_stops
from code.all import SaliencePMI
from code.all import jaccard
from code.all import greedy_macdonald
from code.all import load_fn
from scripts.preprocess_civic import preprocess
#from klm.query import LM, get_unigram_probs
from code.all import score
#from code.wellformedness import SlorScorer

def debug_pmi(K, idea, subset_ids):

    print("Top K={} for idea = {}".format(K, idea))

    for i in pmi.rank_V_by_pmi(subset_ids)[0:K]:
        print(i)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='McDonald Greedy Summarizer')

    parser.add_argument('-b', dest='b', type=int, default=None)
    # parser.add_argument('-K', dest='K', type=int, default=K)
    parser.add_argument("--verbose", dest='verbose', action='store_true')
    parser.add_argument("-idea", dest="idea")
    parser.add_argument("-data", dest="data")
    '''
       There are 2 datasets in the corpora - ccrit and envcanada.
    '''
    args = parser.parse_args()

    if args.idea is None:
        assert "you need" == "to specify an idea"

    scorer = None

    print(args)

    stop_words = get_stops()

    spacy_data = "corpora/data_" + args.data + ".spacy.jsonl"

    preprocess(args.data)
    comments = load_fn(spacy_data)

    subset_ids = [o["docid"] for o in comments if o["idea"] == args.idea]

    start_time =  datetime.datetime.now()
    pmi = PMI(comments, stop_words)
    smi = SaliencePMI(pmi, context_ids=subset_ids, stops=stop_words)
    end_time = datetime.datetime.now()
    print("[*] Computing salience took {}".format((end_time - start_time).total_seconds()))

    subset_docs = [o for o in comments if o["idea"] == args.idea]

    '''
        Setting K based on the number of comments in the idea
    '''

    K = math.ceil(math.log2(len(subset_docs)))

    if args.verbose:
        debug_pmi(K, args.idea, subset_ids)

    start_time =  datetime.datetime.now()
    sum_ = greedy_macdonald(K=K, textual_units=subset_docs,
                            f_salience=smi.salience,
                            f_redundancy=jaccard,
                            scorer=None
                            )
    end_time =  datetime.datetime.now()

    print("***")
    for s in sum_:
        # print(s["presentation_text"])
        print(str(s["presentation_text_short"]) + "...")

    delta = (end_time - start_time)

    print("[*] total time={} seconds, N docs={}".format(delta.total_seconds(), len(subset_docs)))
