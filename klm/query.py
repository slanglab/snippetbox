from __future__ import division

import kenlm
import json

# for corpus formatting see-> https://kheafield.com/code/kenlm/estimation/ esp on <s> and </s> tags
# https://github.com/kpu/kenlm/blob/master/python/example.py. what's up w/ <s> and </s>

def get_unigram_probs(unigram_loc):
    with open(unigram_loc, "r") as inf:
        unigram_log_probs = json.load(inf)
        unigram_log_probs = {k:float(v) for k,v in unigram_log_probs.items()}
        return unigram_log_probs


def slor(sequence, lm, unigram_log_probs_):
    # SLOR function from Jey Han Lau, Alexander Clark, and Shalom Lappin

    sequence = sequence.lower()
    words = sequence.lower().split(" ")
    p_u = sum(unigram_log_probs_[u] for u in words if u in unigram_log_probs_.keys())
    p_u += sum(unigram_log_probs_['<unk>'] for u in words if u not in unigram_log_probs_.keys())

    p_m = lm.score(sequence)

    len_s = len(words) + 0.0

    return (p_m - p_u)/len_s


class LM:

    def __init__(self, loc):

        self.model = kenlm.LanguageModel(loc)

    def score(self, str_, verbose=False):
        # str_ is a " "-delimited string, e.g. "I am a student"
        words = ['<s>'] + str_.split() + ['</s>']


        if verbose:
            for i, (prob, length, oov) in enumerate(self.model.full_scores(str_)):
                print('{0} {1}: {2}'.format(prob, length, ' '.join(words[i+2-length:i+2])))
                if oov:
                    print('\t"{0}" is an OOV'.format(words[i+1]))


        return self.model.score(str_)


if __name__ == "__main__":

    from environments.envs import ENVIRONMENTS
    import argparse
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-e', '--environment', type=str, default="DEV") 

    args = parser.parse_args()

    env = ENVIRONMENTS[args.environment.upper()]
    LOC = env['klm_model']
    UG_MODEL = env["ug_model"]

    lm = LM(loc=LOC)
    up = get_unigram_probs(UG_MODEL)

    print("[*] slor: I am a student")
    print(slor("I am a student", lm, up))

    print("[*] slor: I a student")
    print(slor("I a student", lm, up))
    
    print("[*] slor: student I am")
    print(slor("student i am", lm, up))


    print("[*] slor: American troops")
    print(slor("American troops", lm, up))

    print("[*] slor: American troops landed")
    print(slor("American troops landed", lm, up))

    print("[*] slor: American troops gave")
    print(slor("American troops gave", lm, up))

    print("[*] slor: American troops gave gifts")
    print(slor("American troops gave gifts", lm, up))
