import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm
import math
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances


# This class is just convenient for getting two lists as inputs of the calibration metric computation
class Retrieval:

    def __init__(self, model, data, UIDX, TOPK, feature, test_set) -> None:
        self.model = model
        self.data = data
        self.UIDX = UIDX
        self.TOPK = TOPK
        self.feature = feature # 'category', 'story', ...
        self.test_set = test_set

    def convert_idx_id_train_set(self):
        # conversion between idx and id for train set
        rating_mat = self.model.train_set.matrix
        user_id2idx = self.model.train_set.uid_map
        user_idx2id = list(self.model.train_set.user_ids)
        item_id2idx = self.model.train_set.iid_map
        item_idx2id = list(self.model.train_set.item_ids)

        return rating_mat, user_id2idx, user_idx2id, item_id2idx, item_idx2id

    def convert_idx_id_test_set(self):
        # conversion between idx and id for test set
        rating_mat = self.test_set.matrix
        user_id2idx = self.test_set.uid_map
        user_idx2id = list(self.test_set.user_ids)
        item_id2idx = self.test_set.iid_map
        item_idx2id = list(self.test_set.item_ids)

        return rating_mat, user_id2idx, user_idx2id, item_id2idx, item_idx2id

    def get_recy(self):
        recommendations_idx = self.model.rank(self.UIDX)[0]
        item_idx2id = self.convert_idx_id_train_set()[4]
        recommendations = [item_idx2id[idx] for idx in recommendations_idx[:self.TOPK]]
        dict = self.data.set_index(['item_id'])[self.feature].to_dict()
        recy_list = [dict.get(item) for item in recommendations]

        return recy_list

    def get_history(self):
        item_idx2id = self.convert_idx_id_train_set()[4]
        rating_mat = self.convert_idx_id_train_set()[0]
        rating_arr = rating_mat[self.UIDX].A.ravel()
        # top_rated_items = np.argsort(rating_arr)
        # history = [item_idx2id[i] for i in top_rated_items if i<len(item_idx2id)][:self.TOPK]
        # hist_df = self.data.loc[self.data['item_id'].isin(history)]
        top_rated_items = np.argsort(rating_arr)[-self.TOPK:]
        history = [item_idx2id[i] for i in top_rated_items]
        hist_df = self.data.loc[self.data['item_id'].isin(history)]
        dict = hist_df.set_index(['item_id'])[self.feature].to_dict()
        # user_id = self.convert_idx_id()[1][self.UIDX]
        # history = list(self.data[self.data['user_id']==user_id]['item_id'])
        user_history = [dict.get(item) for item in history]

        return user_history


# Compute diversity metric
class Diversity:
    """Diversity Metric

    Parameters
    ----------
    k: int or list, optional, default: -1 (all)
    The number of items in the top@k list.
    If None, all items will be considered.

    list1: list
    a list of features

    list2: list
    a list of features

    Returns
    ----------
    A number: the result of the metric

    """

    def __init__(self, list1, list2, k=-1):
        self.list1 = list1[:k]
        self.list2 = list2[:k]

    def harmonic_number(self, n):
        """Returns an approximate value of n-th harmonic number.
        http://en.wikipedia.org/wiki/Harmonic_number
        """
        # Euler-Mascheroni constant
        gamma = 0.57721566490153286060651209008240243104215933593992
        return gamma + math.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)

    def compute_distr(self, items, discount=True):
        """Compute the categorical distribution from a given list of Items.

        Parameters
        ----------
        items: list of features

        Returns
        -------
        dist [dict]: dictionary of feature distributions

        """
        n = len(items)
        sum_one_over_ranks = self.harmonic_number(n)
        # In mathematics, the n-th harmonic number is the sum of the reciprocals of the first n natural numbers:
        count = 0
        distr = {}
        # if (discount):
        #   distr = dict((i, arr.count(i)) for i in arr)
        # else:
        #   distr = dict((i, arr.count(i)) for i in arr)

        for indx, item in enumerate(items):
            rank = indx + 1
            story_freq = distr.get(item, 0.)
            distr[item] = story_freq + 1 * 1 / rank / sum_one_over_ranks if discount else story_freq + 1 * 1 / n
            count += 1

        return distr


    def compute_kl_divergence(self, s, q, alpha=0.001):
        """
          params: s, q - two distributions (dict)

          KL (p || q), the lower the better.
          alpha is not really a tuning parameter, it's just there to make the
          computation more numerically stable.
        """
        try:
          assert 0.99 <= sum(s.values()) <= 1.01
          assert 0.99 <= sum(q.values()) <= 1.01
        except AssertionError:
          print("Assertion Error")
          pass
        kl_div = 0.
        ss = []
        qq = []
        merged_dic = self.opt_merge_max_mappings(s, q)
        for key in sorted(merged_dic.keys()):
          q_score = q.get(key, 0.)
          s_score = s.get(key, 0.)
          ss.append((1 - alpha) * s_score + alpha * q_score) # avoid misspecified metrics
          qq.append((1 - alpha) * q_score + alpha * s_score) # avoid misspecified metrics
        kl = entropy(ss, qq, base=2)
        return kl

    def opt_merge_max_mappings(self, dict1, dict2):
        """ Merges two dictionaries based on the largest value in a given mapping.
        Parameters
        ----------
        dict1 : Dict[Any, Comparable]
        dict2 : Dict[Any, Comparable]
        Returns
        -------
        Dict[Any, Comparable]
            The merged dictionary
        """
        # we will iterate over `other` to populate `merged`
        merged, other = (dict1, dict2) if len(dict1) > len(dict2) else (dict2, dict1)
        merged = merged
        for key in other:
          if key not in merged or other[key] > merged[key]:
            merged[key] = other[key]
        return merged

    def compute(self):
        s = self.compute_distr(self.list1)
        q = self.compute_distr(self.list2)

        return self.compute_kl_divergence(s, q)

class Rank:

    def __init__(self, item_rank, item_scores):
        self.item_rank = item_rank
        self.item_scores = item_scores


    def ranking_eval(
            self,
            model,
            metrics,
            train_set,
            test_set,
            val_set=None,
            rating_threshold=0.5,
            exclude_unknowns=True,
            verbose=False,
    ):
        """Evaluate model on provided ranking metrics.

        Parameters
        ----------
        model: :obj:`cornac.models.Recommender`, required
            Recommender model to be evaluated.

        metrics: :obj:`iterable`, required
            List of rating metrics :obj:`cornac.metrics.RankingMetric`.

        train_set: :obj:`cornac.data.Dataset`, required
            Dataset to be used for model training. This will be used to exclude
            observations already appeared during training.

        test_set: :obj:`cornac.data.Dataset`, required
            Dataset to be used for evaluation.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            Dataset to be used for model selection. This will be used to exclude
            observations already appeared during validation.

        rating_threshold: float, optional, default: 1.0
            The threshold to convert ratings into positive or negative feedback.

        exclude_unknowns: bool, optional, default: True
            Ignore unknown users and items during evaluation.

        verbose: bool, optional, default: False
            Output evaluation progress.

        Returns
        -------
        res: (List, List)
            Tuple of two lists:
             - average result for each of the metrics
             - average result per user for each of the metrics

        """

        if len(metrics) == 0:
            return [], []

        avg_results = []
        user_results = [{} for _ in enumerate(metrics)]

        gt_mat = test_set.csr_matrix
        train_mat = train_set.csr_matrix
        val_mat = None if val_set is None else val_set.csr_matrix

        def pos_items(csr_row):
            return [
                item_idx
                for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
                if rating >= rating_threshold
            ]

        for user_idx in tqdm(
                test_set.user_indices, desc="Ranking", disable=not verbose, miniters=100
        ):
            test_pos_items = pos_items(gt_mat.getrow(user_idx))
            if len(test_pos_items) == 0:
                continue

            u_gt_pos = np.zeros(test_set.num_items, dtype=int)
            u_gt_pos[test_pos_items] = 1

            val_pos_items = [] if val_mat is None else pos_items(val_mat.getrow(user_idx))
            train_pos_items = (
                []
                if train_set.is_unk_user(user_idx)
                else pos_items(train_mat.getrow(user_idx))
            )

            u_gt_neg = np.ones(test_set.num_items, dtype=int)
            u_gt_neg[test_pos_items + val_pos_items + train_pos_items] = 0

            # item_indices = None if exclude_unknowns else np.arange(test_set.num_items)
            # item_rank, item_scores = model.rank(user_idx, item_indices)

            for i, mt in enumerate(metrics):
                mt_score = mt.compute(
                    gt_pos=u_gt_pos,
                    gt_neg=u_gt_neg,
                    pd_rank=self.item_rank,
                    pd_scores=self.item_scores,
                )
                user_results[i][user_idx] = mt_score

        # avg results of ranking metrics
        for i, mt in enumerate(metrics):
            avg_results.append(sum(user_results[i].values()) / len(user_results[i]))

        return avg_results

class IntraDiverse:

    def __init__(self):
        pass

    def _single_user_distance(self, predicted, distance_type="cosine"):
        """
        Computes the intra-list distance for a single user's recommendations(feature vector).
        Parameters
        ----------
        predicted : an array of predicted item features.
            Example:
            X = np.array([[2, 3], [3, 5], [5, 8]])

        Returns:
        -------
        ils_single_user: float
            The intra-list distance for a single user's recommendations.
        """
        # Another possible option is first calculate similarity for all items in list.
        # similarity = cosine_similarity(X=recs_content, dense_output=False)
        # diversity = 1 - similarity

        distance = cosine_distances(X=predicted)
        # if distance_type == "pearson":
        #   distance = cdist(X.T, y.T, metric='correlation')

        # get indicies for upper right triangle w/o diagonal
        ##Because the matrix lower left triangle are repeated values.
        upper_right = np.triu_indices(distance.shape[0], k=1)

        # calculate average distance of a list of recommended item features
        ils_single_user = np.mean(distance[upper_right])
        return ils_single_user