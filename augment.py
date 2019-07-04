import os
import pickle
from typing import Sequence, Tuple, List

import numpy as np
from collections import OrderedDict, deque, defaultdict

from tqdm import tqdm

from tts.search import VectorSimilaritySearch
from tts.utils import flatten_result


class EmbeddingLoader:
    def __init__(self, embedding_dir, embed_dim=512):
        self.embedding_dir = embedding_dir
        self.embedding_splits = [(name, self.parse_embed_name(name)) for name in self.embedding_dir]
        self.embedding_splits.sort(key=lambda x: x[1][0])
        self.embed_dim = embed_dim

    def get_bucket(self, doc_id):
        for i, (name, (s, e)) in enumerate(self.embedding_splits):
            if s <= doc_id <= e:
                return i

    def load(self, queries):
        query_ids = [(self.get_bucket(idx), i) for i, idx in enumerate(queries)]

        split_to_ids = OrderedDict()
        for split_id, q_id in query_ids:
            split_to_ids.setdefault(split_id, []).append(q_id)

        all_embeds = np.zeros(shape=(len(queries), self.embed_dim), dtype=np.float32)
        last_index = 0

        buckets = sorted(list(split_to_ids.keys()))
        embed_ids_to_query_ids = deque()
        for b_id in buckets:
            q_ids = buckets[b_id]

            split_name, (s, e) = self.embedding_splits[b_id]
            emb_split = np.load(os.path.join(self.embedding_dir, split_name))

            for q_id in zip(q_ids):
                embed_ids_to_query_ids.append(q_id)
                query = queries[q_id]

                local_query = query - s

                all_embeds[last_index, :] = emb_split[local_query]
                last_index += 1

        embed_ids_to_query_ids = [(q_id, embed_id) for embed_id, q_id in enumerate(embed_ids_to_query_ids)]
        embed_ids_to_query_ids.sort(key=lambda x: x[0])

        assert np.all(np.equal(
            np.array(list(map(lambda x: x[0], embed_ids_to_query_ids))),
            np.array(list(map(lambda x: x[1], query_ids)))
        ))

        embed_ids = [embed_id for q_id, embed_id in embed_ids_to_query_ids]

        all_embeds = all_embeds[embed_ids, :]

        return all_embeds

    @staticmethod
    def parse_embed_name(name):
        splts = name.split('.npy')[0].split('_')
        return int(splts[1]), int(splts[3])


docs = []
docs_ner = []

JAC_LOWER_LIM = 0.45
JAC_UPPER_LIM = 0.65
PQ_JAC_LOWER_LIM = 0.7

NUM_PQS = 3


def start_and_stop_index(lst, x):
    first = lst.index(x)
    last = len(lst) - lst[::-1].index(x) - 1
    return first, last


def jaccard(s1, s2):
    return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))


def filter_pairs(pairs):
    valid_pairs = []
    for x_id, y_id in pairs:
        x = docs_ner[x_id]
        y = docs_ner[y_id]

        x_words = x.split()
        y_words = y.split()

        jaccard_sim = jaccard(set(x_words), set(y_words))
        if JAC_LOWER_LIM <= jaccard_sim <= JAC_UPPER_LIM:
            valid_pairs.append((x_id, y_id))

    inverse_valid_pairs = [(y, x) for x, y in valid_pairs]
    valid_pairs += inverse_valid_pairs
    valid_pairs = list(set(valid_pairs))
    valid_pairs = np.array(valid_pairs)

    return valid_pairs


def filter_bad_xpq(xy_pqs, doc_mapping, pair_mapping):
    valid_tuples = deque()
    for xy_id, pq_id in xy_pqs:
        x_id = doc_mapping[xy_id]
        p_id, q_id = pair_mapping[pq_id]

        if x_id == p_id or x_id == q_id:
            continue

        x = docs_ner[x_id]
        p = docs_ner[p_id]
        q = docs_ner[q_id]

        if x == p or x == q:
            continue

        valid_tuples.append((x_id, (p_id, q_id)))

    return valid_tuples


def select_pqs(xy, pqs):
    x_id, _ = xy
    x_words = docs_ner[x_id].split()

    jac_scores = []
    for p_id, q_id in pqs:
        p_words = docs_ner[p_id].split()

        jac_score = jaccard(set(x_words), set(p_words))
        if jac_score > PQ_JAC_LOWER_LIM:
            jac_scores.append(((p_id, q_id), jac_score))

    jac_scores.sort(key=lambda x: x[1], reverse=True)

    pqs = [pq for pq, score in jac_scores][:NUM_PQS]

    return pqs


def aggregate_and_create_plan(xy_pq_tuples) -> Sequence[Tuple[int, Sequence[Tuple[int, int]]]]:
    x_to_pqs = defaultdict(list)
    for x, pq in xy_pq_tuples:
        x_to_pqs[x].append(pq)

    plans = []
    for xy in x_to_pqs.keys():
        x_id = xy
        plans.append((x_id, select_pqs(xy, x_to_pqs[xy])))

    return x_to_pqs


def flatten(xy_to_pqs: Sequence[Tuple[int, Sequence[Tuple[int, int]]]]) -> Sequence[List[int]]:
    flt = deque()
    for (x, pqs) in xy_to_pqs:
        flt_pqs = []
        for p, q in pqs:
            flt_pqs += [p, q]

        # flt_pqs = [x, p1, q1, p2, q2, ...]
        flt_pqs = [x] + flt_pqs

        flt.append(flt_pqs)

    return flt


def augment(episode_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes = pickle.load(open(episode_path, 'rb'))
    embedding_loader = EmbeddingLoader('emb')

    plan_len = 1 + NUM_PQS * 2

    all_plans = deque()
    all_plan_classes = deque()
    plans_to_episode_ids = deque()

    for eps_id, eps in enumerate(episodes):
        print("#", eps_id + 1, '/', len(episodes))
        d_train, d_test, d_classes = eps
        d_train_ids = [doc_id for doc_id, class_id in d_train]
        doc_id_to_dtrain_id = {d: i for i, d in enumerate(d_train_ids)}
        embeds = embedding_loader.load(d_train_ids)
        print("Loaded Embeds")

        ss = VectorSimilaritySearch(embeds, num_candidate_neighbors=16, num_actual_neighbors=10, use_gpu=True)
        pairs = ss.create_pair_dataset()
        pairs = [(d_train_ids[x_id], d_train_ids[y_id]) for x_id, y_id in pairs]
        pairs = filter_pairs(pairs)
        pairs_x = pairs[:, 0]
        pairs_x_embed = embedding_loader.load(pairs_x)
        del ss
        print("Created pairs, len(pairs) =", len(pairs))

        ss = VectorSimilaritySearch(pairs_x_embed, num_candidate_neighbors=16, num_actual_neighbors=10, use_gpu=True)
        xpq_pairs = ss.search(embeds)
        xpq_pairs = flatten_result(xpq_pairs)
        xpq_pairs = filter_bad_xpq(xpq_pairs, d_train_ids, pairs)
        del ss
        print("Created xpq_pairs, len(xpq_pairs) =", len(xpq_pairs))

        x_to_pqs = aggregate_and_create_plan(xpq_pairs)
        plans = flatten(x_to_pqs)
        plans = list(filter(lambda x: len(x) > 1, plans))

        for p in plans:
            padding = [-1] * (plan_len - len(p))
            all_plans.append(p + padding)
            all_plan_classes.append(d_train[doc_id_to_dtrain_id[p[0]]][1])

        print("len(plans) =", plans)

        plans_to_episode_ids.extend([eps_id] * len(plans))

    plans_to_episode_ids = np.array(list(plans_to_episode_ids))
    all_plan_classes = np.array(list(all_plan_classes))
    all_plans = np.array(all_plans, dtype=np.int)

    assert all_plans.shape[-1] == plan_len
    assert len(all_plans) == len(plans_to_episode_ids) == len(all_plan_classes)

    return all_plans, plans_to_episode_ids, all_plan_classes


if __name__ == '__main__':
    augment()
