import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm
import math

class MovielensRetrieval:

    def __init__(self, model, item_df, UIDX, TOPK, test_set) -> None:
        self.model = model
        self.item_df = item_df
        self.UIDX = UIDX
        self.TOPK = TOPK
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
        item_idx2id = self.convert_idx_id_train_set()[4]
        recommendations = self.model.rank(self.UIDX)[0]
        distr = self.item_df.loc[[int(item_idx2id[i]) for i in recommendations[:self.TOPK]]]
        dict = {}
        for index, row in distr.items():
            dict[index] = np.count_nonzero(np.array(row))
        del dict['Title']
        del dict['Release Date']

        return dict

    def get_history(self):
        item_idx2id = self.convert_idx_id_train_set()[4]
        rating_mat = self.convert_idx_id_train_set()[0]
        rating_arr = rating_mat[self.UIDX].A.ravel()
        # top_rated_items = np.argsort(rating_arr)
        # history = [item_idx2id[i] for i in top_rated_items if i < len(item_idx2id)][:self.TOPK]
        # distr = self.item_df.loc[[int(h) for h in history]]
        top_rated_items = np.argsort(rating_arr)[-self.TOPK:]
        distr = self.item_df.loc[[int(item_idx2id[i]) for i in top_rated_items]]
        dict = {}
        for index, row in distr.items():
            dict[index] = np.count_nonzero(np.array(row))
        del dict['Title']
        del dict['Release Date']

        return dict

class Diversity:

    def __init__(self):
        pass

    def compute_dict_distr(self,distr):
        """Compute the categorical distribution from a given list of Items.

        Returns
        -------
        dist [dict]: dictionary of feature distributions

        """

        # normalize the summed up probability, so it sums up to 1
        total_n = sum(list(distr.values()))
        for item, count in distr.items():
            distr[item] = count / total_n

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
