
import numpy as np


class RerankingAlgorithm:
    """Re-ranking algorithm

    Attributes
    ------
    k: int or list, optional, default: 0
        The number of items in the top@k list.
        If 0, all items stay the same.

    lambda_constant: float
        weight factor of the diversity metrics.

    diversity_objective :obj:{`<cornac.metrics.DiversityMetric>`}, required
        A collection of metrics to use to evaluate the recommender models, \
        e.g., [Calibration].

    """

    def __init__(self, k=0, rerank=0, lambda_constant=0,
                 gt_ratings=None, diversity_objective=None,
                 u_gt_pos=None, u_gt_neg=None, rating_threshold=None,
                 globalProbs_dict=None, user_history=None, pool_ids=None,
                 pd_other_users_dict=None, user_idx=None):

        self.k = k
        self.rerank = rerank
        self.lambda_constant = lambda_constant
        self.diversity_objective = diversity_objective,
        self.gt_ratings = gt_ratings  # gd relevance value
        self.u_gt_pos = u_gt_pos
        self.u_gt_neg = u_gt_neg
        self.rating_threshold = rating_threshold
        self.globalProbs_dict = globalProbs_dict
        self.user_history = user_history
        self.pool_ids = pool_ids
        self.pd_other_users_dict = pd_other_users_dict
        self.user_idx = user_idx

    def div(self, item_rank_new, item_scores_new):
        """Compute diversity metrics for new item list

        Parameters
        -------
        item_rank_new: list
            New item ranking list

        item_scores_new: list
            New ranked item scores list

        Returns
        -------
        The newly computed diversity metric results
        """
        scores = []
        for mt in self.diversity_objective[0]:
            if "Binomial" in mt.name:
                globalProbs = self.globalProbs_dict["Binomial"]
            else:
                globalProbs = []
            if "Fragmentation" in mt.name:
                pd_other_users = self.pd_other_users_dict["Fragmentation"]
            else:
                pd_other_users = []
            mt_score = mt.compute(
                gt_pos=self.u_gt_pos,
                gt_neg=self.u_gt_neg,
                pd_rank=item_rank_new,
                pd_scores=item_scores_new,
                rating_threshold=self.rating_threshold,
                gt_ratings=self.gt_ratings,  # gd relevance value
                globalProb=globalProbs,
                user_history=self.user_history,
                pool=self.pool_ids,
                pd_other_users=pd_other_users
            )
            if mt_score is None:
                scores.append(-1)
            else:
                scores.append(mt_score)
        return sum(scores)

    def re_rank(self, itemScoreDict):
        """Re-rank the recommendation list using greedy algorithm

        Parameters
        -------
        itemScoreDict: dict
            The dictionary for items and their relative predicted scores
            E.g., {item_ids: pre_score}

        Returns
        -------
        New recommendation list after re-ranking
        """

        s, score, r = [], [], list(itemScoreDict.keys())

        if self.k == 0 or self.rerank == 0 or self.rerank > len(r):
            return r

        while len(r) > 0:
            max = 0
            selectOne = None
            score = [itemScoreDict[m] for m in s]
            for i in r:
                sim1 = itemScoreDict[i]
                sim2 = 0
                candidate = s + [i]
                candidate_score = score + [itemScoreDict[i]]
                div = self.div(candidate, candidate_score)
                if div <= sim2:
                    continue
                elif div > sim2:
                    sim2 = div
                curr_value = self.lambda_constant * sim1 - (1 - self.lambda_constant) * sim2
                if abs(curr_value) > max:  # compare absolute value
                    max = abs(curr_value)
                    selectOne = i

            if selectOne == None:
                selectOne = i
            r.remove(selectOne)
            s.append(selectOne)

            if len(s) == self.rerank:
                s += r
                break
        return s
