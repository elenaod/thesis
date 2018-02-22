""" Main function. """

from util import is_word, load_model_embedding, info_from_line, log, get_tensor, repack_tensors
from data import InputData
from nn import CNN

import time
from gensim.models import KeyedVectors
from functools import partial
import tensorflow
from tensorflow.python import debug

def filter_1(string):
    return is_word(string)

MODEL = KeyedVectors.load_word2vec_format("../data/GoogleNews-vectors-negative300.bin",
                                          binary=True)
    
def readFile(path):
    load_model = partial(load_model_embedding, model=MODEL)
    inp = InputData(path, info_from_line, load_model, [filter_1])
    return inp

def print_info_map(map_name, map_data, top_k, file):
    with open(file, 'w') as f:
        for k, v in sorted(map_data.items(), key=lambda x: x[1], reverse=True)[0:top_k]:
            f.write("'%s'\t%d\n" % (k, v))

def create_file_info(path, prefix="", save_file_pref="", data_f=None, model_f=None):
    inp = readFile(path)

    all_words = {}
    wrong_words = {}
    mentions = {}
    mention_distances = []

    documents = None
    if prefix:
        documents = inp.get_documents_by_prefix(prefix)
    else:
        documents = inp.get_all_documents()
    for d in documents:
        last_mention = 0
        last_mention_sentence = 0
        for u in d.children:
            for s in u.children:
                for w in s.children:
                    words = w.children
                    for word in words:
                        if word in all_words.keys():
                            all_words[word] = all_words[word] + 1
                        else:
                            all_words[word] = 1
                    
                        if word not in MODEL.vocab.keys():
                            if word in wrong_words.keys():
                                wrong_words[word] = wrong_words[word] + 1
                            else:
                                wrong_words[word] = 1
                    if w.mentionId:
                        mention_distances.append((last_mention, last_mention_sentence))
                        last_mention = 0
                        last_mention_sentence = 0
                        if " ".join(words) in mentions.keys():
                            mentions[" ".join(words)] = mentions[" ".join(words)] + 1
                        else:
                            mentions[" ".join(words)] = 1
                    else:
                        last_mention = last_mention + 1
            last_mention_sentence = last_mention_sentence + 1

    print_info_map("MISSPELLED WORDS", wrong_words, len(wrong_words), "f:/thesis/wrong_words_data_%s.txt" % save_file_pref)
    print_info_map("MENTIONS", mentions, len(mentions), "f:/thesis/mentions_data_%s.txt" % save_file_pref)
    print_info_map("ALL WORDS", all_words, len(mentions), "f:/thesis/all_words_data.%s.txt" % save_file_pref)

    total_word_count = sum(all_words.values())
    print("total words read: %d; embeddings not found for %d" % \
          (total_word_count, sum(wrong_words.values())))
    print("unique words found: %d, unique mispelled %d" %
           (len(all_words), len(wrong_words)))
    print("unique mentions: %d, total mentions: %d" %( len(mentions), sum(mentions.values())))
    print("average mention distance in words: %0.3f" % (sum([x[0] for x in mention_distances]) / len(mention_distances)))
    print("average mention distance in words: %0.3f" % (sum([x[1] for x in mention_distances]) / len(mention_distances)))
    sorted_md = sorted(mention_distances, key=lambda x: x[0])
    print("max mention_distance %s, min mention distance: %s" % (sorted_md[0][0], sorted_md[-1][0]))

def create_and_train(file, emb_features, d_features, mention_pair_features):
    nn = CNN(
        embedding_features=emb_features,
        semantic_features=d_features,
        mention_pair_features=mention_pair_features,
        optimizer=tensorflow.train.RMSPropOptimizer(learning_rate=0.95)
    )
    nn.create_nn()

    inp = readFile(file)
    test_inp = readFile("../data/friends.test.scene_delim.conll.txt")

    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    session = tensorflow.Session(config=config)
    session.run(tensorflow.global_variables_initializer())
    writer = tensorflow.summary.FileWriter("f:/thesis/vals/log/", tensorflow.get_default_graph())
    saver = tensorflow.train.Saver()
    save_path = saver.save(session, "f:/thesis/vals/models/model-0.ckpt")
    train_writer = tensorflow.summary.FileWriter('f:/thesis/vals/summary/train',
                                                 session.graph)

    documents = inp.get_documents_by_prefix("(/friends-s01e01); part 003")

    iteration = 1
    for d in documents:
        mentions = d.get_mentions()
        pairs = list(zip(mentions[:-1], mentions[1:]))
        mention_pairs = [mp for mp in pairs]
        labels = [(1.0 if mp[0].mentionId == mp[1].mentionId else 0.0) for mp in pairs]
        t_s = time.time()
        for pair, label in zip(pairs, labels):
            nn.train_nn(session,
                        mention_pair=pair,
                        label=label,
                        writer=train_writer)
        t_e = time.time()
        log("info",
            "Trained %d mentions in %0.3f min" % (len(mention_pairs), (t_e - t_s) / 60))
        save_path = saver.save(session, "f:/thesis/tmp/models/model-%d.ckpt" % iteration)
        iteration = iteration + 1

    documents = test_inp.get_documents_by_prefix("(/friends-s01e01)")
    for d in documents:
        mentions = d.get_mentions()
        mention_pairs = [mp for mp in zip(mentions[:-1], mentions[1:])]
        t_s = time.time()
        for pair in mention_pairs:
            print(pair)
            res = nn.evaluate(session, mention_pair=pair)
            log("info", "NN predict: %s vs label: %s" % \
                (res, pair[0].mentionId == pair[1].mentionId))
        t_e = time.time()
        log("info",
            "Predicted %d mentions in %0.3f min" % (len(mention_pairs), (t_e - t_s) / 60))
    return nn

def mention_emb(mention):
    return tensorflow.stack(
        [tensorflow.reduce_mean(mention.embeddings, axis=0, keep_dims=True)]
    )

def surrounding_word(mention):
    return tensorflow.stack(
        [tensorflow.reduce_mean(mention.embeddings, axis=0, keep_dims=True)] +
        [tensorflow.reduce_mean(get_tensor(w, shape=[1, 1, 300]), axis=0, keep_dims=True) for w in mention.get_prev(3)] +
        [tensorflow.reduce_mean(get_tensor(w, shape=[1, 1, 300]), axis=0, keep_dims=True) for w in mention.get_next(3)],
        axis=1
    )

def surrounding_sentence(mention):
    sent = mention.parent
    return tensorflow.stack(
        [tensorflow.reduce_mean(mention.embeddings, axis=0, keep_dims=True)] +
        [tensorflow.reduce_mean(get_tensor(w, shape=[1, 1, 300]), axis=0, keep_dims=True) for w in sent.get_prev(3)] +
        [tensorflow.reduce_mean(get_tensor(w, shape=[1, 1, 300]), axis=0, keep_dims=True) for w in sent.get_next(3)],
        axis=1
    )

def surrounding_utterance(mention):
    u = mention.parent.parent
    return tensorflow.stack(
        [tensorflow.reduce_mean(u.embeddings, axis=0, keep_dims=True)] +
        [tensorflow.reduce_mean(get_tensor(w), axis=0, keep_dims=True) for w in u.get_prev(3)] +
        [tensorflow.reduce_mean(get_tensor(w), axis=0, keep_dims=True) for w in u.get_next(1)],
        axis=1
    )

def gender_info(mention):
    return tensorflow.zeros(shape=[1, 1, 1, 5], dtype=tensorflow.float32)

def mention_dist_info(mention_pair):
    speaker_match = (mention_pair[0].parent.parent.speaker == mention_pair[1].parent.parent.speaker)
    return tensorflow.reshape(
        tensorflow.constant(0 if speaker_match else 1, dtype=tensorflow.float32),
        shape=[1, 1, 1, 1]
    )

EMBEDDING_FEATURES = [
    {
        "func": mention_emb,
        "name": "mention_embedding",
        "kernel": 1,
        "shape": [1, 1, 1, 300]
    },
    {
        "func": surrounding_word,
        "name": "surrounding_word",
        "kernel": 7,
        "shape": [1, 7, 1, 300]
    },
    {
        "func": surrounding_sentence,
        "name": "surrounding_sentence",
        "kernel": 7,
        "shape": [1, 7, 1, 300]
    },
    {
        "func": surrounding_utterance,
        "name": "surrounding_utterance",
        "kernel": 5,
        "shape": [1, 5, 1, 300]
    },
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
        "shape": [1, 1, 1, 1]
    }
]

create_and_train("../data/friends.train.scene_delim.conll",
                 EMBEDDING_FEATURES,
                 SEMANTIC_FEATURES,
                 MENTION_PAIR_FEATURES)

#inp = readFile("../data/friends.train.scene_delim.conll")
#for d in inp.get_all_documents("(/friends-s01e01)"):
#    m = d.get_mentions()[0]
#    print(m.embeddings, m.parent.embeddings)
#    for e in EMBEDDING_FEATURES[:4]:
#        print(e["name"], e["func"](m))
#print(len(mentions)) # 13268
#print(len(mentions)) # 2430
