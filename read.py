"""
Reads the data from a CoNLL file.

The data we work with is grouped into (in order of size, from largest to
smallest:
    - utterances - consists of one or more sentences
    - sentences - consists of one or more words
    - words - that is the smallest piece of data that we work with.

Most of the transformations that we need are applied first to the words and
then the values we require for phrases/sentences/utterances are computed from
the base values for words.

"""

from itertools import combinations

# TODO: move tensorflow out of this file
import tensorflow
from autocorrect import spell
from nltk import edit_distance

import const
from util import log, split_line, is_word, not_matching, strip_symbols, \
                 EMBEDDING_SHAPE, stack_tensors, get_tensor

def main_character_heuristic(character_key):
    return "_and_" not in character_key and \
           "'" not in character_key and \
           all([s[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for s in character_key.split("_")])

def compare_distances(dist):
    print("Calculating distances for %s" % dist)
    if 0 in dist:
        return 0
    else:
        return sum(dist)

def calculate_distances(character_key, phrase):
    if phrase in character_key:
        return character_key, 0
    for i in range(len(phrase), 1, -1):
        if phrase[:i] in character_key[:i] or \
           phrase[len(phrase) - i:] in character_key[len(character_key) - i:]:
            return character_key, len(phrase) - i 
    return (character_key, len(character_key))


def determine_mention_id_named_characters(word, characters):
    """
    If we have a character name in the mention, or the character is
    referring to themselves, then assign the appropriate mention_id.
    """
    if word.mentionId > 0:
        return word.mentionId
    # TODO: use distance to character names?
    # TODO: still not working with "Pheebs"
    pronouns = ["you", "he", "she", "it", "we", "they"]
    for p in pronouns:
        if p in word.children:
            return word.mentionId
    if 'I' in word.children:
        return get_speaker_embedding(characters,
                                     word.parent.parent.speaker)
    if word.ner == "PERSON":
        # We are not really interested in multiple mentions:
        interesting_names = [k for k in characters.keys() if main_character_heuristic(k)]
        print(interesting_names)
        all_characters = [calculate_distances(v, "_".join(word.children)) \
                          for v in interesting_names]
        sorted_characters = sorted(all_characters,
                                   key=lambda x: x[1])
        log("info", "SIEVE_1: Mapping mention of %s to mention id %s" % \
                    (sorted_characters[0][0], " ".join(word.children)))
        return get_speaker_embedding(characters, sorted_characters[0][0])


class LLElement():
    """
    Base class for a linked list element. Not sure if neeeded.
    Actually, I could probably move a lot of the code here if I use higher
    order functions. That's for February, if I get the time.
    """
    def __init__(self, parent=None, prev=None, _next=None):
        self.parent = parent
        self.prev = prev
        self.next = _next
        self.children = []
        self.embeddings = None

    def get_parent(self):
        return self.parent or LLElement()

    def get_prev(self, n=0):
        prev_els = [None] * n
        i = 0
        c = self
        while c and c.prev and i < n:
            prev_els[i] = c.prev
            c = c.prev
        return prev_els

    def get_next(self, n=0):
        next_els = [None] * n
        i = 0
        c = self
        while c and c.prev and i < n:
            next_els[i] = c.prev
            c = c.prev
        return next_els

    def get_mentions(self):
        mentions = []
        for child in self.children:
            mentions = mentions + child.get_mentions()
        return mentions

def load_speaker_embeddings(speaker_key_file):
    characters = {}

    with open(speaker_key_file, 'r') as f:
        lines = f.read().split("\n")[:-1]
        for line in lines:
            key, *values = split_line(line)
            characters["_".join(values)] = key

    return characters

def get_speaker_embedding(characters, speaker_key):
    try:
        return characters[speaker_key]
    except KeyError:
        log("warn", "No character with key %s found!" % speaker_key)
        return -1

# TODO: consequtive words with the same mention ID?
class Word(LLElement):
    def __init__(self, parent, lines, prev):
        # also add lemmatizer - should cut down on the warnings, at least
        LLElement.__init__(self, parent, prev)
        input_data =self.parent.parent.parent.parent
        self.children = [l["inf"] for l in lines \
                        if all([f(l["inf"]) for f in input_data.filters])]
        assert None not in self.children, "invalid child found!"
        self.children = [w for w in self.children if w]
        self.ner = strip_symbols("".join([l["ner_tag"] for l in lines]))
        self.mentionId = strip_symbols(lines[0]["mention_id"])
        self.mentionId = None if self.mentionId == "-" else int(self.mentionId)
        if len(self.children) > 0:
            self.embeddings = [[load_embedding(l) for load_embedding in input_data.wordEmbeddings] \
                                for l in self.children]
            self._notFound = any([any([e[1] for e in emb]) for emb in self.embeddings])
        else:
            self.embeddings = [[get_tensor(None)]]
            self._notFound = False
        self.embeddings = tensorflow.stack(
            [tensorflow.stack([e[0] for e in emb]) for emb in self.embeddings]
        )

    def make_word(self, string):
        return strip_symbols(string)

    def get_avg_embedding(self, idxs):
        # idxs are indexes of the words from the phrase that should be considered
        return tensorflow.reduce_mean(self.embeddings, axis=1)

    def get_mentions(self):
        return [self] if self.mentionId else []

    def get_word_tensor(self):
        idxs = range(0, len(self.children))
        if len(self.children) == 0:
            return tensorflow.zeros([1, 300], dtype=tensorflow.float32)
        if "oh" in " ".join(self.children):
            return tensorflow.zeros([1, 300], dtype=tensorflow.float32)
        tensors = []
        # TODO: turn this into a list comprehension?
        for i in idxs:
            igrams = list(combinations(idxs, i + 1))
            tensors = tensors + [self.get_avg_embedding(igram) for igram in igrams]
        # there's 2047, 1, 300 which is disturbing...
        return tensorflow.reduce_mean(tensorflow.stack(tensors), axis=0)

# TODO: what to do with empty sentences?

# TODO: Need better name

class Sentence(LLElement):
    def __init__(self, utterance, lines, prev):
        LLElement.__init__(self, utterance, prev)
        self.split_into_words(self, lines)
        assert None not in self.children, "invalid child found!"
        if len(self.children) > 1:
            self.embeddings = tensorflow.concat([c.embeddings for c in self.children], axis=0)
        elif len(self.children) == 1:
            self.embeddings = self.children[0].embeddings
        else:
            self.embeddings = get_tensor(None, shape=[1, 1, 300])

    def split_into_words(self, sentence, lines):
        i, j = 0, 0
        prev_word = None
        while i < len(lines):
            ner = "".join([l["ner_tag"] for l in lines[i:j + 1]])
            mention = "".join([l["mention_id"] for l in lines[j:i + 1]])
            while j < len(lines) and (not_matching(ner) or not_matching(mention)):
                j = j + 1
                ner = "".join([l["ner_tag"] for l in lines[i:j + 1]])
                mention = "".join([l["mention_id"] for l in lines[j:i + 1]])
            new_word = Word(self, lines[i:j + 1], prev_word)
            if prev_word:
                prev_word.next = new_word
            self.children.append(new_word)
            prev_word = new_word
            i, j = j + 1, j + 1

class Utterance(LLElement):
    def __init__(self, episode, speaker, lines, prev, parent):
        LLElement.__init__(self, parent, prev)
        self.speaker = speaker
        self.episode = episode
        self.split_into_sentences(lines)
        assert None not in self.children, "invalid child found!"
        if len(self.children) > 1:
            self.embeddings = tensorflow.concat([c.embeddings for c in self.children], axis=0)
        elif len(self.children) == 1:
            self.embeddings = self.children[0].embeddings
        else:
            self.embeddings = get_tensor(None, shape=[1, 1, 300])

    def create_sentence(self, lines):
        new_sentence = Sentence(self, lines, self.prev_sentence)
        if self.prev_sentence:
            self.prev_sentence.next = new_sentence
        self.prev_sentence = new_sentence
        self.children.append(new_sentence)

    def split_into_sentences(self, lines):
        i, j = 0, 0
        self.prev_sentence = None 
        while None in lines[i:]:
            j = lines.index(None, i)
            self.create_sentence(lines[i:j])
            i = j + 1
        self.create_sentence(lines[i:])

class Document():
    def __init__(self, lines, idx, parent):
        self.__name = idx
        self.parent = parent
        self.children = []
        self._extract_utterances(lines)
        log("info", "Read document %s" % self.__name)

    def _extract_utterances(self, lines):
        self._prev_utterance = None
        i, j = 0, lines.index(None, 0)
        current_episode, current_speaker = lines[i]["filename"], lines[i]["speaker"]
        while j + 1 < len(lines):
            new_episode, new_speaker = lines[j+1]["filename"], lines[j+1]["speaker"]
            if (new_speaker, new_episode) != (current_speaker, current_episode):
                self.create_utterance(current_episode,
                                      current_speaker,
                                      lines[i:j - 1])
                current_speaker, current_episode = new_speaker, new_episode
                i = j + 1
            j = lines.index(None, j + 1)
        # was j - 1
        self.create_utterance(current_episode, current_speaker, lines[i:j])

    def create_utterance(self, current_episode, current_speaker, lines):
        # Those should be read from somewhere?
        new_utterance = Utterance(current_episode, current_speaker,
                                  lines, self._prev_utterance, self)
        self.children.append(new_utterance)
        if self._prev_utterance:
            self._prev_utterance.next = new_utterance
        self._prev_utterance = new_utterance

    def get_mentions(self):
        mentions = []
        for u in self.children:
            mentions = mentions + u.get_mentions()
        return mentions

class InputData():
    """
    Class to represent the input of a character identification task.
    Requires a single word and all associated annotations for it to be on one line.
    Required annotations for every word:
        - filename - the name of the file where the word occurs
        - pos_tag - the POS tag of the word
        - sent_tag - the sentence tag for the word
        - inf - the inifinitve of the word
        - speaker - the speaker, if the word is part of someone's speech
        - ner_tag - the NER tag of the word
        - mention_id - the ID of the entity which this word references.
    """
    def __init__(self, filename, colInfo, wordEmbeddings=None, filters=None):
        """
            Constructs an InputData object.
            @param: filename -- the name of the file containing the data
            @param: colInfo -- a function to extract annotations from a line
            @param: wordEmebddings -- a list of word embeddings to be used.
        """
        log("info", "Reading from file %s" % filename)
        self._filename = filename
        self._columnInfo = colInfo
        self.wordEmbeddings = wordEmbeddings
        self.filters = filters
        self._document_data = []
        with open(self._filename, 'r') as f:
            self._extract_common(f)

    def get_all_documents(self):
        return (
            Document([self._columnInfo(l) for l in s["lines"]], s["name"], self) \
                     for s in self._document_data
        )

    def get_documents_by_prefix(self, prefix):
        return (
            Document([self._columnInfo(l) for l in s["lines"]], s["name"], self) \
            for s in self._document_data if s["name"].startswith(prefix)
        )

    def _extract_common(self, f):
        lines = f.read().split("\n")
        docs_begin = [lines.index(l) for l in lines if l.startswith("#begin")]
        docs_end = [l[0] for l in enumerate(lines) if l[1].startswith("#end")]
        assert len(docs_begin) == len(docs_end), \
               "found %d tags to begin document and %d tags to end document" % \
               (len(docs_begin), len(docs_end))
        log("info", "%d documents found in input data" % len(docs_begin))
        for d in zip(docs_begin, docs_end):
            doc_name = " ".join(lines[d[0]].split(" ")[2:])
            self._document_data.append({"lines": lines[d[0] + 1:d[1]], "name": doc_name})
