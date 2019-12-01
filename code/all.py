from stop_words import get_stop_words
import json
import string
import numpy as np

def load_fn(fn):
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
    stop_words = stop_words + ["San", "Diego", "El", "Nudillo", "plastic"]
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

        Note: if the textual unit already has a cached salience field then
              just go and return this without recomputing. Other salience
              functions should also support

        '''
        if "salience" in textual_unit:
            return textual_unit["salience"]

        pmis = [self.pmi.compute_pmi(o, self.context_ids) for o in textual_unit["tokens"] if o.lower() not in self.stops]
        if len(pmis) <= 3: # avoid weird short sentences
            salience = 0
        else:
            pmis.sort(reverse=True)
            # pmi for top 5 tokens. Otherwise it penalizes long sentences
            # b/c many sentences have lots of words w/ min PMI
            salience = np.sum(pmis[0:5])/len(pmis[0:5])
        textual_unit["salience"]  = salience
        return textual_unit["salience"]

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


def greedy_macdonald(K, textual_units, f_salience, f_redundancy, N=1000, scorer=None, verbose=False):
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

        add_this["print_as"] = " ".join(add_this["tokens"])

        summary = summary + [add_this]

    return summary
