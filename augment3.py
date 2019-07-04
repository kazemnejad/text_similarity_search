import pickle
from collections import deque, defaultdict
from typing import Sequence, Tuple, List

import numpy as np

from tts.search import MinHashSimilaritySearch
from tts.utils import flatten_result

docs = []
docs_ner = []
docs_lemma = []

JAC_LOWER_LIM = 0.45
JAC_UPPER_LIM = 0.65
PQ_JAC_LOWER_LIM = 0.7
N_THREADS = 16

NUM_PQS = 3


def load_from_list(indices, lst):
    return [lst[i] for i in indices]


def start_and_stop_index(lst, x):
    first = lst.index(x)
    last = len(lst) - lst[::-1].index(x) - 1
    return first, last


def jaccard(s1, s2):
    return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))


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
        # if jac_score > PQ_JAC_LOWER_LIM:
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

    return plans


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


def augment(episode_path: str, pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes = pickle.load(open(episode_path, 'rb'))
    plan_len = 1 + NUM_PQS * 2

    all_plans = deque()
    all_plan_classes = deque()
    plans_to_episode_ids = deque()

    for eps_id, eps in enumerate(episodes):
        print("#", eps_id + 1, '/', len(episodes))
        d_train, d_test, d_classes = eps
        d_train_ids = [doc_id for doc_id, class_id in d_train]
        doc_id_to_dtrain_id = {d: i for i, d in enumerate(d_train_ids)}

        d_train_str = load_from_list(d_train_ids, docs)
        d_train_lemma = load_from_list(d_train_ids, docs_lemma)

        xpq_pairs = pairs_ss.search((d_train_lemma, d_train_str))
        xpq_pairs = flatten_result(xpq_pairs)
        xpq_pairs = filter_bad_xpq(xpq_pairs, d_train_ids, pairs)
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
    pairs_path = ''

    pairs = np.load(pairs_path)
    pairs_x = pairs[:, 0]

    pairs_x_str = load_from_list(pairs_x, docs)
    pairs_x_lemma = load_from_list(pairs_x, docs_lemma)

    pairs_ss = MinHashSimilaritySearch(pairs_x_str, pairs_x_lemma, num_candidate_neighbors=16, num_actual_neighbors=10,
                                       jaccard_threshold=JAC_LOWER_LIM, batch_size=200)
