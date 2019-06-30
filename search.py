import glob
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

import minhash_funcs
import utils


class SimilaritySearch:
    def __init__(self, source_data, **kwargs):
        pass

    def search(self, query):
        pass

    def create_pair_dataset(self):
        pass

    def save_search_result(self, result: Sequence[Sequence[int]], output_path: str):
        flatten = utils.flatten_result(result)
        np.save(output_path, flatten)

    def save_pair_dataset(self, pairs: np.ndarray, output_path: str):
        np.save(output_path, pairs)

    @staticmethod
    def read_source_data(filepath):
        pass

    @staticmethod
    def read_query(filepath):
        pass


class VectorSimilaritySearch(SimilaritySearch):
    def __init__(self, source_data: np.ndarray, batch_size=512, num_candidate_neighbors=20, num_actual_neighbors=10,
                 select_mode='best', remove_self_result=True, metric='IP', use_gpu=False, **kwargs):
        super().__init__(source_data)

        self.source_data = source_data
        self.use_gpu = use_gpu
        self.metric = metric
        self.batch_size = batch_size
        self.num_candidate_neighbors = num_candidate_neighbors
        self.num_actual_neighbors = num_actual_neighbors
        self.select_mode = select_mode
        self.remove_self_result = remove_self_result

        self._init()

    def _init(self):
        import faiss

        self.dim = self.source_data.shape[-1]

        if self.use_gpu:
            self.gpu_res = faiss.StandardGpuResources()
            if self.metric == 'IP':
                self.index = faiss.GpuIndexFlatIP(self.gpu_res, self.dim)
            elif self.metric == 'L2':
                self.index = faiss.GpuIndexFlatL2(self.gpu_res, self.dim)

        else:
            if self.metric == 'IP':
                self.index = faiss.IndexFlatIP(self.dim)
            elif self.metric == 'L2':
                self.index = faiss.IndexFlatL2(self.dim)

        self.index.add(self.source_data)

    def _do_search(self, query):
        result = []
        for batch_index, batch_query in tqdm(enumerate(utils.chunks(query, self.batch_size)),
                                             total=len(query) // self.batch_size):
            _, batch_result = self.index.search(batch_query, self.num_candidate_neighbors)

            for i in range(len(batch_query)):
                q_index_in_dataset = batch_index * self.batch_size + i
                candidates = batch_result[i]
                neighbors = self._select_from_candidates(q_index_in_dataset, candidates)

                result.append(neighbors)

        assert len(query) == len(result)

        return result

    def _select_from_candidates(self, q_index_in_dataset, candidates):
        if self.remove_self_result:
            valid_candidates = candidates[candidates != q_index_in_dataset]
        else:
            valid_candidates = candidates

        if self.select_mode == 'best':
            return valid_candidates[:self.num_actual_neighbors]
        elif self.select_mode == 'sample':
            return np.random.choice(valid_candidates, self.num_actual_neighbors, replace=False)
        else:
            raise ValueError("select_mode \in ['sample', 'best']")

    def search(self, query):
        return self._do_search(query)

    def create_pair_dataset(self):
        result = self.search(self.source_data)
        pairs = utils.flatten_result(result)
        return pairs

    @staticmethod
    def read_source_data(filepath):
        return np.load(filepath)

    @staticmethod
    def read_query(filepath):
        return np.load(filepath)


class MinHashSimilaritySearch(SimilaritySearch):
    def __init__(self, source_data, source_data_lemma, num_candidate_neighbors=20, num_actual_neighbors=10,
                 num_threads=4, remove_self_result=False, select_mode='best', jaccard_threshold=0.5, **kwargs):
        super().__init__(source_data_lemma)

        self.source_data_lemma = source_data_lemma
        self.sentence_mapping = source_data
        self.num_threads = num_threads
        self.jaccard_threshold = jaccard_threshold
        self.num_candidate_neighbors = num_candidate_neighbors
        self.num_actual_neighbors = num_actual_neighbors
        self.remove_self_result = remove_self_result
        self.select_mode = select_mode

        self._init()

    def _init(self):
        self.tmp_dir = Path('working_tmp')
        if self.tmp_dir.exists():
            shutil.rmtree(str(self.tmp_dir))

        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        lemma_with_ids = []
        for i, doc in enumerate(self.source_data_lemma):
            lemma_with_ids.append(
                '%s:%s' % (i, doc)
            )

        self.source_data_lemma = lemma_with_ids

        print("Num instances: %s" % len(self.source_data_lemma))

        self.lsh_filenames = self._make_lsh(str(self.tmp_dir / 'lsh_lem'))

    def _make_lsh(self, filepath: str = 'lsh') -> Sequence[str]:
        data_lemma_path = str(self.tmp_dir / 'data_lemma.txt')
        utils.save_list(self.source_data_lemma, data_lemma_path)

        lsh_filenames = minhash_funcs.parallel_make_lsh(
            data_lemma_path,
            filepath,
            self.num_threads, batch_size=len(self.source_data_lemma) // self.num_threads,
            thresh=self.jaccard_threshold
        )

        return lsh_filenames

    def _do_search(self, query_lemma, query):
        query_lemma_toks = [q.split(' ') for q in query_lemma]
        query_keys = list(range(len(query)))

        result = minhash_funcs.lsh_query_parallel(
            self.lsh_filenames,
            query_lemma_toks,
            query_keys,
            nproc=self.num_threads,
            jac_tr=self.jaccard_threshold
        )

        for k, v in result.items():
            result[k] = [int(minhash_funcs.find_sent_id(d)) for d in v]

        final_result = []
        for i, q_str in enumerate(query):
            q_words = set(q_str.split())

            candidates = result[i]
            jaccard_scores = [
                (
                    c_id,
                    minhash_funcs.jaccard(q_words, set(self.sentence_mapping[c_id].split()))
                ) for c_id in candidates]
            jaccard_scores = list(filter(lambda x: x[1] > self.jaccard_threshold, jaccard_scores))
            jaccard_scores.sort(key=lambda x: x[1], reverse=True)

            candidates = np.array([c_id for c_id, _ in jaccard_scores])
            neighbors = self._select_from_candidates(q_str, candidates)
            final_result.append(neighbors)

        return final_result

    def _select_from_candidates(self, q_str, candidates) -> np.ndarray:
        if self.remove_self_result:
            is_not_equal_to_q = [q_str != self.sentence_mapping[c_id] for c_id in candidates]
            valid_candidates = candidates[is_not_equal_to_q]
        else:
            valid_candidates = candidates

        if self.select_mode == 'best':
            return valid_candidates[:self.num_actual_neighbors]
        elif self.select_mode == 'sample':
            if len(valid_candidates) <= self.num_actual_neighbors:
                return valid_candidates
            else:
                return np.random.choice(valid_candidates, self.num_actual_neighbors, replace=False)
        else:
            raise ValueError("select_mode \in ['sample', 'best']")

    def search(self, query):
        return np.array(self._do_search(*query))

    def create_pair_dataset(self) -> np.ndarray:
        seed = self.source_data_lemma
        minhash_funcs.generate_adjlist(
            seed,
            len(self.source_data_lemma) // self.num_threads,
            str(self.tmp_dir / 'splitadj'),
            self.lsh_filenames,
            nproc=self.num_threads,
            jac_tr=self.jaccard_threshold
        )

        pairs = minhash_funcs.make_ids_pairs(
            str(self.tmp_dir / 'splitadj_adjlist.txt'),
            5 * 1e5,
            self.sentence_mapping,
            lower_jac=self.jaccard_threshold,
            maxbucket=self.num_actual_neighbors
        )
        pairs = np.array(pairs, dtype=np.int)

        assert len(pairs.shape) == 2

        return pairs

    def clean(self):
        shutil.rmtree(str(self.tmp_dir))

    @staticmethod
    def read_source_data(filepath):
        source_data = utils.read_list(filepath)
        source_data_lemma = utils.read_list('%s_lemma' % filepath)

        return source_data, source_data_lemma

    @staticmethod
    def read_query(filepath):
        query = utils.read_list(filepath)
        query_lemma = utils.read_list('%s_lemma' % filepath)

        return query_lemma, query


def test_vector_similarity_search():
    d = 8  # dimension
    nb = 10  # database size
    np.random.seed(1234)  # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xq = xb

    vss = VectorSimilaritySearch(xb, batch_size=200, metric="L2", num_candidate_neighbors=4, num_actual_neighbors=2,
                                 select_mode='sample')
    I = vss.search(xq)

    print(I)


def test_minhash_similarity_search():
    source_data = utils.read_list('source_data.txt')
    source_data_lemma = utils.read_list('source_data.txt_lemma')

    mss = MinHashSimilaritySearch(source_data, source_data_lemma,
                                  num_candidate_neighbors=20, num_actual_neighbors=10,
                                  num_threads=4, remove_self_result=True,
                                  select_mode='best', jaccard_threshold=0.4)

    pairs = mss.create_pair_dataset()
    pairs_str = [(source_data[i], source_data[j]) for i, j in pairs]

    from pprint import pprint
    pprint(pairs_str)


def test_minhash_similarity_search_query():
    source_data = utils.read_list('source_data.txt')
    source_data_lemma = utils.read_list('source_data.txt_lemma')

    query = source_data[:10]
    query_lemma = source_data_lemma[:10]

    mss = MinHashSimilaritySearch(source_data, source_data_lemma,
                                  num_candidate_neighbors=20, num_actual_neighbors=10,
                                  num_threads=4, remove_self_result=True,
                                  select_mode='best', jaccard_threshold=0.2)

    results = mss.search((query_lemma, query))
    # pairs_str = [(source_data[i], source_data[j]) for i, j in pairs]

    from pprint import pprint
    pprint(results)


if __name__ == '__main__':
    # test_vector_similarity_search()
    test_minhash_similarity_search()
    # test_minhash_similarity_search_query()
