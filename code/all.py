from stop_words import get_stop_words
import json
from code.wellformedness import shorten_sentence
import string
import numpy as np

def load_fn(fn = "corpora/data_ccrit.spacy.jsonl"):
    stop_words = get_stops()
    comments = []
    with open(fn, "r") as inf:
        for i in inf:
            i = json.loads(i)
            comments.append(i)
    return comments


def get_stops():
    stop_words = get_stop_words('en')
    stop_words = stop_words + [o for o in string.punctuation] 
    stop_words = stop_words + ["'s", "nt", "n't", ' ']
    stop_words = stop_words + ["San", "Diego", "El", "Nudillo"]
    return [i.lower() for i in stop_words]


def jaccard(s1,s2):
    s1t = set(s1["tokens"])
    s2t = set(s2["tokens"])
    return len(s1t & s2t) / len(s1t | s2t)


def score(summary, f_salience, f_redundancy):
    '''
    Equation 1, McDonald. Score a summary. Note f_salience = "Rel" and
                                            f_redundancy = "Red"

    Assumes that K is satisfied

    summary is a list of textual units = [{"tokens":{"fountain", "big", "homeless"}}
                                           ...
                                          {"tokens":{"waste", "foundtain", "water"}}]


    '''
    score = 0
    for i, unit_i in enumerate(summary):
        score += f_salience(unit_i)
        for j in range(0, i):
            score -= f_redundancy(summary[j], unit_i)
    return score


class SaliencePMI(object):
    '''
    compute salience based on PMI in a given context

    Heuristically penalize short comments.
    '''

    def __init__(self, pmi, context_ids, stops):
        self.pmi = pmi
        self.context_ids = context_ids
        self.stops = stops

    def salience(self, textual_unit):
        '''
        Salience operates on a textual_unit (e.g. sentence or paragraph)
            - For community crit, its comments
        '''
        pmis = [self.pmi.compute_pmi(o, self.context_ids) for o in textual_unit["tokens"] if o.lower() not in self.stops]
        if len(pmis) <= 3: # avoid weird short sentences
            return 0
        else:
            pmis.sort(reverse=True)
            # pmi for top 5 tokens. Otherwise it penalizes long sentences 
            # b/c many sentences have lots of words w/ min PMI
            return np.sum(pmis[0:5])/len(pmis)


def get_ranked_textual_units(units, f_salience):

    for i in units:
        i["salience"] = f_salience(i)

    units.sort(key=lambda x: x["salience"], reverse=True)

    return units


def stupid_sum(K, textual_units, f_salience, f_redundancy):
    '''
    Anytime summarization algorithm.

    Assume 2 sets of textual units:
        those in your summary | those not = all sentences

    At each iter:
        pick a s1 from those not propto a distribution
        pick a s2 from the summary propto a distribution
        is the summary better off w/ s2 than s1? If so, replace it


    Speed matters more than anything.
    '''
    pass


def greedy_macdonald(K, textual_units, f_salience, f_redundancy, b=None, N=1000, scorer=None, verbose=False):
    '''
    Greedy approximation from the McDonald paper. It's pretty similar to sumbasic

    interestingly: this algo is quadratic. It makes K passes thru the list
    '''
    ranked_units_remaining = get_ranked_textual_units(textual_units, f_salience)
    summary = []


    while len(summary) < K:
        max_so_far = -100000000000000

        # safety checks
        if len(summary) == len(textual_units):
            break
        if ranked_units_remaining == []:
            break

        add_this = None

        for s in ranked_units_remaining:
            score_for_this = score(summary + [s], f_salience, f_redundancy)
            if score_for_this > max_so_far:
                add_this = s 
                max_so_far = score_for_this
                if verbose:
                    print(add_this["tokens"], score_for_this)

        if add_this is not None:
            ranked_units_remaining.remove(add_this)

        if b is not None:
            print_as = shorten_sentence(add_this, scorer, b=b, N=N)
            add_this["print_as"] = print_as
        else:
            add_this["print_as"] = " ".join(add_this["tokens"])

        summary = summary + [add_this]

    return summary
