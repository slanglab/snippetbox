'''
Methods for computing salience functions
'''

import numpy as np


class pmiSalience(object):
    '''
    Sample Use
    helper = pmiHelper(docs)

    Each doc is a dictionary with "tokens" and a "docid"

    a = helper.compute_pmi(word='fountain', context=[1,5,7,9]) where context is docids
    see https://web.stanford.edu/~jurafsky/slp3/6.pdf on PMI
    '''
    def __init__(self, docs):

        V = set(v for doc in docs for v in doc["tokens"])

        self.v2n = {v: k for k, v in enumerate(V)}

        # map from docno to columnno in TDM
        self.context2n = {v["docno"]: k for k, v in enumerate(docs)}

        tdm = np.zeros((len(V), len(docs)))

        # plus one smoothing to fix some issues w/ PMI (see jurafsky book)
        tdm += 1

        for docno, doc in enumerate(docs):
            toks = doc["tokens"]
            for tok in toks:
                tdm[self.tok2n[tok]][doc["docid"]] += 1

        self.grand_total = np.sum(tdm)
        self.tdm = tdm

    def compute_pmi(self, word, context):
        '''
        note: here context is defined across documents

        note: here context is a list of docids
        '''

        context_column_indexes = [self.context2n[o] for o in context]
        word_index = self.v2n[word]

        count_word_context = np.sum(self.tdm[word_index][context_column_indexes])
        p_word_context = count_word_context / self.grand_total
        p_word = np.sum(self.tdm[word_index])/self.grand_total
        p_context = np.sum(self.tdm[:, context_column_indexes])/self.grand_total

        return np.log(p_word_context/(p_word * p_context))