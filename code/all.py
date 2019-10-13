from stop_words import get_stop_words

def get_stops():
    stop_words = get_stop_words('en')
    stop_words = stop_words + [o for o in string.punctuation] 
    stop_words = stop_words + ["'s", "nt", "n't", ' ']
    stop_words = stop_words + ["San", "Diego", "El", "Nudillo"]
    return stop-words
