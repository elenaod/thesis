""" Main function. """

from util import is_word, load_model_embedding, info_from_line
from read import InputData
from nn import CNN

from gensim.models import KeyedVectors
from functools import partial
import tensorflow

def filter_1(string):
    return is_word(string)

def readFile(path):
    model = KeyedVectors.load_word2vec_format("../data/GoogleNews-vectors-negative300.bin",
                                              binary=True)
    load_model = partial(load_model_embedding, model=model)
    inp = InputData(path, info_from_line, [load_model], [filter_1])
#    speaker_embeddings = read.load_speaker_embeddings("../data/friends_entity_map.txt")
    return inp

def print_info_map(map_name, map_data, top_k):
    print(map_name)
    print("======================")
    for k, v in sorted(map_data.items(), key=lambda x: x[1], reverse=True)[0:top_k]:
        print("     %s -> %d" % (k, v))
    print("======================")

def create_file_info(path):
    inp = readFile(path)

    words = {}
    wrong_words = {}
    mentions = {}
    mention_distances = []
    last_mention = 0
    last_mention_sentence = 0

    for d in inp.documents:
        for u in d.children:
            print("utterance embedding", u.embeddings)
            for s in u.children:
                print("sentence embedding", s.embeddings)
                for w in s.children:
                    print("word embedding", w.embeddings)
                    word = " ".join(w.children)
                    if word in words.keys():
                        words[word] = words[word] + 1
                    else:
                        words[word] = 1
                    if w._notFound:
                        if word in wrong_words.keys():
                            wrong_words[word] = wrong_words[word] + 1
                        else:
                            wrong_words[word] = 1
                    if w.mentionId:
                        mention_distances.append((last_mention, last_mention_sentence))
                        last_mention = 0
                        last_mention_sentence = 0
                        if word in mentions.keys():
                            mentions[word] = mentions[word] + 1
                        else:
                            mentions[word] = 1
                    else:
                        last_mention = last_mention + 1
            last_mention_sentence = last_mention_sentence + 1

    print_info_map("MISSPELLED WORDS", wrong_words, 100)
    print_info_map("MENTIONS", mentions, 100)
    print_info_map("ALL WORDS", words, 100)

    total_word_count = sum(words.values())
    print("total words read: %d; embeddings not found for %d" % \
          (total_word_count, sum(wrong_words.values())))
    print("unique words found: %d, unique mispelled %d" %
           (len(words), len(wrong_words)))
    print("unique mentions: %d", len(mentions))

def create_and_train(file, emb_features, d_features, mention_pair_features):
    nn = CNN(
        embedding_features=emb_features,
        semantic_features=d_features,
        mention_pair_features=mention_pair_features
    )
    nn.create_nn()

    inp = readFile(file)

    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    session = tensorflow.Session(config=config)
    session.run(tensorflow.global_variables_initializer())

    idx = 0
    for d in inp.documents:
        mentions = d.get_mentions()
        mention_pairs = [mp for mp in zip(mentions[:-1], mentions[1:])]
        i = 0
        for pair in mention_pairs:
            nn.train_nn(session,
                        mention_pair=pair,
                        label=(1.0 if pair[0].mentionId == pair[1].mentionId else 0.0))
        if i % 500 == 0:
            print("Trained %s/%s mentions" % (i, len(mention_pairs)))
        idx = idx + 1
        if idx > 2:
            break
    return nn

def mention_emb(mention):
    return tensorflow.stack(
        [tensorflow.reduce_mean(mention.embeddings, axis=0, keep_dims=True)]
    )

def gender_info(mention):
    return tensorflow.zeros(shape=[1, 1, 1, 5], dtype=tensorflow.float32)

def mention_dist_info(mention_pair):
    return tensorflow.zeros(shape=[1, 1, 1, 3], dtype=tensorflow.float32)

EMBEDDING_FEATURES = [
    {
        "func": mention_emb,
        "name": "mention_embedding",
        "kernel": 1,
        "shape": [1, 1, 1, 300]
    }
]

SEMANTIC_FEATURES = [
    {
        "func": gender_info,
        "name": "gender_info",
        "shape": [1, 1, 1, 5]
    }
]

MENTION_PAIR_FEATURES = [
    {
        "func": mention_dist_info,
        "name": "mention_dist",
        "shape": [1, 1, 1, 3]
    }
]

create_and_train("../data/friends.train.scene_delim.conll",
                 EMBEDDING_FEATURES,
                 SEMANTIC_FEATURES,
                 MENTION_PAIR_FEATURES)

# create_file_info('../data/friends.train.scene_delim.conll')
# create_file_info('../data/friends.test.scene_delim.conll.nokey')