from stop_words import get_stop_words
import string
import numpy as np

def get_stops():
    stop_words = get_stop_words('en')
    stop_words = stop_words + [o for o in string.punctuation] 
    stop_words = stop_words + ["'s", "nt", "n't", ' ']
    stop_words = stop_words + ["San", "Diego", "El", "Nudillo"]
    return stop_words


class SaliencePMI(object):
    '''compute salience based on PMI in a given context'''
    
    def __init__(self, pmi, context_ids):
        self.pmi = pmi
        self.context_ids = context_ids
    
    def salience(self, textual_unit):
        '''
        Salience operates on a textual_unit (e.g. sentence or paragraph)
            - For community crit, its comments
        '''
        pmis = [self.pmi.compute_pmi(o, self.context_ids) for o in textual_unit["tokens"]]
        return np.mean(pmis)
