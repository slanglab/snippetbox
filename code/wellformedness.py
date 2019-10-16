import json
import numpy as np
import re
from random import choice
from klm.query import LM, get_unigram_probs
from code.all import get_stops
from klm.query import slor

def print_me(ix, o):
    return o if ix == 1 else "..."

def clean_dots(str_):
    a = "[... ]{2,8}"
    b = "\.{6,}"
    fix_multi_dots = re.sub(a, "...", str_)
    remove_long_dots = re.sub(b, "...", fix_multi_dots)
    return remove_long_dots


class SlorScorer(object):
    
    def __init__(self, lm, up):
        self.lm = lm
        self.up = up
    
    def min_slor_scorer(self, input_snippet):
        ### See evernote: how do you use SLOR to make a good snippet?
        all_ = []
        for seq in input_snippet.split("..."):
            all_.append(slor(seq, self.lm, self.up))
        return min(all_)


def shorten_sentence(comment, scorer, b=100, N=1000):

    ops = [(" ".join(comment["tokens"])[0:b], scorer(" ".join(comment["tokens"])[0:b]))] # default answer
    
    for oo in range(N):

        included = np.ones(len(comment["tokens"]))

        for o in range(choice([1,2])):

            d1 = choice(range(len(comment["tokens"])))
            included[d1] = 0
            deleted = choice(range(10))

            for i in range(deleted):
                if i + d1 < len(included):
                    included[i + d1] = 0


        comment_snippet = [print_me(ix, o) for ix, o in zip(included, comment["tokens"])]

        sequence = " ".join([o for o in comment_snippet if o != "..."])

        remove_long_dots = clean_dots(" ".join(comment_snippet))
    
        if len(remove_long_dots) < b:
            ops.append((remove_long_dots, scorer(remove_long_dots)))

    return list(set(ops))

if __name__ == "__main__":
    stop_words = get_stops()

    ix = 13
    
    comments = []
    with open("corpora/data_ccrit.spacy.jsonl", "r") as inf:
        for i in inf:
            i = json.loads(i)
            comments.append(i)

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
    ops = shorten_sentence(comments[ix], scorer.min_slor_scorer)
    ops.sort(key=lambda x:x[1], reverse=True)
    print("***")
    print(ops[0:4])
