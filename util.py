""" Some utility functions and variables. """

import re
from os.path import basename
from numpy import load
from gensim.models import KeyedVectors
from autocorrect import spell
from tensorflow import zeros, constant, float32, add, multiply, stack, \
                       reduce_mean, concat

_RE_NUM = re.compile("[0-9]")
_RE_WORD = re.compile("[a-zA-Z]")
_RE_SPECIAL = re.compile("[^0-9a-zA-Z_-]")
_STOPWORDS = ['a', 'to', 'and', 'of'] # stopwords.words("english")

EMBEDDING_SHAPE = [1, 300]

# TODO: do not read the file all at once, we do not need to load it into the
# RAM entirely to extract the data

def log(log_lvl, msg):
    """ simple logging function. """
    print("[%s] [%s] %s" % (basename(__file__), log_lvl, msg))

# The point of is_number and is_word is to enforce a sort of type-checking
# where the types are "ID" and "english word".

def is_number(string):
    """ Helper method to check if a string represents a number/id """
    return bool(_RE_NUM.findall(string)) and not bool(_RE_WORD.findall(string))

def contains_special(string):
    """
    Helper method to check if a string contains any special symbols
    special symbols are ()*,!.?' etc.
    """
    return bool(_RE_SPECIAL.findall(string))

def strip_symbols(string):
    """ Helper method to strip all special symbols from a string. """
    return re.sub(_RE_SPECIAL, "", string)

def is_word(string):
    """
    Helper method to check if a string represents a (valid, English) word.
    do not check for all stopwords - we need verbs?

    TODO: do dictionary check. 3:00 should not pass, because we can't get
    any word-related information from it; also wan-na should not pass.
    We should either correct those on the fly (if they're several edge cases
    which are easily taken care of); or ignore them so that the input data is
    as clean as possible.

    """
    return strip_symbols(string) and \
           bool(_RE_WORD.findall(string)) and \
           not bool(_RE_NUM.findall(string)) #and \
#           bool(strip_symbols(string) not in _STOPWORDS)

def correct_word(string):
    """
    Receives a string and returns a correct dictionary word.
    Used to handle 'okayyy", "intelignect" and so on.
    """
    pass

def not_matching(string):
    """
    Helper method to check if a string contains matching paranthesis.
    Used to check if the word on the current line is part of a phrase.
    """
    return string.count('(') != string.count(')')

def split_line(line):
    """ Extracts space-separated values from a string. """
    return re.sub(r"\s+", " ", line).split(" ")

def mspell(word):
    if word == "ya":
        return "you"
    if word == "yknow":
        return "know"
    return word

def info_from_line(line):
    if line == "":
        return None
    columns = split_line(line)
    if len(columns) < 12:
        log("warn", "Invalid line found: %s, skipping..." % line)
        return None
    return {
        "filename": columns[0],
        "pos_tag": columns[4],
        "sent_tag": columns[5],
        "inf": mspell(columns[6]),
        "speaker": columns[9],
        "ner_tag": columns[10],
        "mention_id": columns[11]
    }

def load_pre_built_embedding(word, data_dir, retry=True, emb_shape=[300]):
    try:
        return constant(load("%s/%s.npy" % (data_dir, word)).tolist(),
                        dtype=float32), True
    except FileNotFoundError:
        log("warn", "Missing embedding for word '%s' in pre-built vocabulary" % word)
        if retry:
            return load_pre_built_embedding(spell(word), data_dir, retry=False)
        else:
            return zeros(shape=emb_shape, dtype=float32), False

def load_model_embedding(word, model, retry=False, emb_shape=[300]):
    try:
        return constant(model.word_vec(word)), True
    except KeyError:
        log("warn", "Missing embedding for word '%s' in model" % word)
        if retry:
            return load_pre_built_embedding(spell(word), model, retry=False)
        else:
            return zeros(shape=emb_shape, dtype=float32), False

def load_embedding(word, data_dir="vocab", model=None, emb_shape=[300]):
    """
    Helper function to load the word embedding from either a file or a model
    """
    if model:
        return load_model_embedding(word, model)
    else:
        return load_pre_built_embedding(word, data_dir)
    
def get_tensor(obj, shape=EMBEDDING_SHAPE):
    """
    Helper function which returns the embedding for either an object or None
    Don't need this - if we get an empty list, set embedding to zeros.
    also pad the lists with some elements before/after delayed eval?
    """
    return  obj.embeddings if obj else zeros(shape=shape, dtype=float32)

def stack_tensors(arr, shape=EMBEDDING_SHAPE):
    if arr == []:
        return zeros(shape=EMBEDDING_SHAPE)
    elif len(arr) == 1:
        return zeros(shape=EMBEDDING_SHAPE)
    else:
        return \
            stack(
                [reduce_mean(
                    concat(
                        [el.embeddings for el in arr],
                        axis=0
                    ),
                    0
                )]
            )

def create_feed_dict(session, placeholder_arr, val_arr):
    """ Helper function to create a feed_dict for session.run """
    d = {}
    for p, v in zip(placeholder_arr, val_arr):
        for p_el, v_el in zip(p, v):
            d[p_el] = session.run(v_el)
    return d

def multiply_and_add(inp, weights, biases):
    """ Helper function to perform w * x + b """
    return add(multiply(inp, weights), biases)