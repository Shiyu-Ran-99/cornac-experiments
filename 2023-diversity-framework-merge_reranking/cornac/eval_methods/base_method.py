# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from collections import OrderedDict
import time

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from ..data import FeatureModality
from ..data import TextModality, ReviewModality
from ..data import ImageModality
from ..data import GraphModality
from ..data import SentimentModality
from ..data import Dataset
from ..metrics import RatingMetric
from ..metrics import RankingMetric
from ..metrics import DiversityMetric
from ..metrics import RerankingAlgorithm
from ..experiment.result import Result
from ..utils import get_rng
from ..utils.common import MyConfigParser


def save_model_parameter(fpath="parameters.ini"):
    config = MyConfigParser()
    file_path = fpath
    config.read(file_path)
    para_info = {}
    for each_section in config.sections():
        para_info[each_section] = config.getlist(
            each_section, 'tune_parameters')
    return para_info


def rating_eval(model, metrics, test_set, user_based=False, verbose=False):
    """Evaluate model on provided rating metrics.

    Parameters
    ----------
    model: :obj:`cornac.models.Recommender`, required
        Recommender model to be evaluated.

    metrics: :obj:`iterable`, required
        List of rating metrics :obj:`cornac.metrics.RatingMetric`.

    test_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for evaluation.

    user_based: bool, optional, default: False
        Evaluation mode. Whether results are averaging based on number of users or number of ratings.

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
    user_results = []

    (u_indices, i_indices, r_values) = test_set.uir_tuple
    r_preds = np.fromiter(
        tqdm(
            (
                model.rate(user_idx, item_idx).item()
                for user_idx, item_idx in zip(u_indices, i_indices)
            ),
            desc="Rating",
            disable=not verbose,
            miniters=100,
            total=len(u_indices),
        ),
        dtype='float',
    )

    gt_mat = test_set.csr_matrix
    pd_mat = csr_matrix((r_preds, (u_indices, i_indices)), shape=gt_mat.shape)

    for mt in metrics:
        if user_based:  # averaging over users
            user_results.append(
                {
                    user_idx: mt.compute(
                        gt_ratings=gt_mat.getrow(user_idx).data,
                        pd_ratings=pd_mat.getrow(user_idx).data,
                    ).item()
                    for user_idx in test_set.user_indices
                }
            )
            avg_results.append(
                sum(user_results[-1].values()) / len(user_results[-1]))
        else:  # averaging over ratings
            user_results.append({})
            avg_results.append(mt.compute(
                gt_ratings=r_values, pd_ratings=r_preds))

    return avg_results, user_results


def ranking_eval(
        model,
        metrics,
        train_set,
        test_set,
        val_set=None,
        rating_threshold=1.0,
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

        u_gt_pos = np.zeros(test_set.num_items, dtype='int')
        u_gt_pos[test_pos_items] = 1

        val_pos_items = [] if val_mat is None else pos_items(
            val_mat.getrow(user_idx))
        train_pos_items = (
            []
            if train_set.is_unk_user(user_idx)
            else pos_items(train_mat.getrow(user_idx))
        )

        u_gt_neg = np.ones(test_set.num_items, dtype='int')
        u_gt_neg[test_pos_items + val_pos_items + train_pos_items] = 0

        item_indices = None if exclude_unknowns else np.arange(
            test_set.num_items)
        item_rank, item_scores = model.rank(user_idx, item_indices)

        for i, mt in enumerate(metrics):
            mt_score = mt.compute(
                gt_pos=u_gt_pos,
                gt_neg=u_gt_neg,
                pd_rank=item_rank,
                pd_scores=item_scores,
            )
            user_results[i][user_idx] = mt_score

     # avg results of ranking metrics
    for i, mt in enumerate(metrics):
        avg_results.append(
            sum(user_results[i].values()) / len(user_results[i]))

    return avg_results, user_results


def diversity_eval(
        model,
        metrics,
        train_set,
        test_set,
        val_set=None,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=False,
):
    """Evaluate model on provided diversity metrics.

    Parameters
    ----------
    model: :obj:`cornac.models.Recommender`, required
        Recommender model to be evaluated.

    metrics: :obj:`iterable`, required
        List of diversity metrics :obj:`cornac.metrics.DiversityMetric`.

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
        return [], [], []

    avg_results = []
    user_results = [{} for _ in enumerate(metrics)]
    user_info = []
    gt_mat = test_set.csr_matrix
    train_mat = train_set.csr_matrix
    val_mat = None if val_set is None else val_set.csr_matrix

    def pos_items(csr_row):
        return [
            item_idx
            for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
            if rating >= rating_threshold
        ]

    def get_gd_ratings(csr_row):
        return [rating
                for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
                ]

    def get_gd_idx(csr_row):
        return [item_idx
                for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
                ]
    evaluated_ids = {}
    evaluated_ids['total_users'] = len(test_set.user_indices)
    user_history_dict = OrderedDict()
    for user_idx in (test_set.user_indices):
        pos_item_idx = (
            []
            if train_set.is_unk_user(user_idx)
            else pos_items(train_mat.getrow(user_idx))
        )
        user_history_dict[user_idx] = pos_item_idx
    for user_idx in tqdm(
            test_set.user_indices, desc="Diversity evaluation", disable=not verbose, miniters=100
    ):
        test_pos_items = pos_items(gt_mat.getrow(
            user_idx))
        if len(test_pos_items) == 0:
            continue

        u_gt_pos = np.zeros(test_set.num_items, dtype='int')
        u_gt_pos[test_pos_items] = 1

        val_pos_items = [] if val_mat is None else pos_items(
            val_mat.getrow(user_idx))
        train_pos_items = (
            []
            if train_set.is_unk_user(user_idx)
            else pos_items(train_mat.getrow(user_idx))
        )

        u_gt_neg = np.ones(test_set.num_items, dtype='int')
        u_gt_neg[test_pos_items + val_pos_items + train_pos_items] = 0

        item_indices = None if exclude_unknowns else np.arange(
            test_set.num_items)
        item_rank, item_scores = model.rank(user_idx, item_indices)
        pool_ids = np.arange(test_set.num_items)
        u_gt_rating = np.zeros(test_set.num_items)
        gd_item_idx = get_gd_idx(gt_mat.getrow(user_idx))
        gd_item_rating = get_gd_ratings(gt_mat.getrow(user_idx))
        for i in range(len(gd_item_idx)):
            # user ground truth rating
            u_gt_rating[gd_item_idx[i]] = gd_item_rating[i]
        # interacted and positive rating in training set
        user_history = user_history_dict[user_idx]
        # check if metrics contain Binomial
        globalProbs = []
        for i, mt in enumerate(metrics):
            if "Binomial" in mt.name:
                global_prob = mt.globalFeatureProbs(user_history_dict)
                globalProbs.append(global_prob)
            else:
                globalProbs.append([])
        pd_other_users = []
        userIdx = list(test_set.user_indices)
        for i, mt in enumerate(metrics):
            if "Fragmentation" in mt.name:
                index = userIdx.index(user_idx)
                candidate_user_list = list(
                    userIdx[0:index]) + list(userIdx[index + 1:])
                samples = list(np.random.choice(
                    candidate_user_list, size=mt.n_samples, replace=False))
                if mt.k > 0:
                    sample_rank = [model.rank(x, item_indices)[
                        0][:mt.k] for x in samples]
                else:
                    sample_rank = [model.rank(x, item_indices)[0]
                                   for x in samples]
                pd_other_users.append(sample_rank)
            else:
                pd_other_users.append([])
        for i, mt in enumerate(metrics):
            mt_score = mt.compute(
                gt_pos=u_gt_pos,
                gt_neg=u_gt_neg,
                pd_rank=item_rank,
                pd_scores=item_scores,
                rating_threshold=rating_threshold,
                gt_ratings=u_gt_rating,
                globalProb=globalProbs[i],
                user_history=user_history,
                pool=pool_ids,
                pd_other_users=pd_other_users[i]
            )

            if mt_score is None:
                pass
            else:
                if mt.name not in evaluated_ids:
                    evaluated_ids[mt.name] = 1
                elif mt.name in evaluated_ids:
                    evaluated_ids[mt.name] = evaluated_ids[mt.name]+1
                user_results[i][user_idx] = mt_score

    for i, mt in enumerate(metrics):
        if len(user_results[i]) > 0:
            avg_results.append(
                sum(user_results[i].values()) / len(user_results[i]))
        else:
            avg_results.append(-1)
            print("No results found for metric {}".format(mt.name))
    for i, mt in enumerate(metrics):
        if mt.name in evaluated_ids:
            user_info.append(evaluated_ids[mt.name])
        else:
            user_info.append(0)
    # last one is total number of users in the test set
    user_info.append(
        evaluated_ids['total_users'])
    return avg_results, user_results, user_info


def reranking_diversity_eval(
        model,
        train_set,
        test_set,
        val_set=None,
        k=0,
        rerank=0,
        lambda_constant=0,
        diversity_objective=None,
        reranking_metrics=None,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=False,
):
    """Evaluate model on provided diversity metrics.

    Parameters
    ----------
    model: :obj:`cornac.models.Recommender`, required
        Recommender model to be evaluated.

    k: int
        This parameter is only useful if you are considering re-ranking.
        e.g., 20, the top 20 items recommended will be reranked.

    rerank: int
        This parameter is only useful if you are considering re-ranking.
        e.g., 200, the top @k items recommended to be reranked will be chosen from top 200 items.

    lambda_constant: float
        weight factor of the diversity metrics.

    diversity_objective :obj:{`<cornac.metrics.DiversityMetric>`}, required
        A collection of metrics to use to evaluate the recommender models for reranking, \
        e.g., [Calibration].

    reranking_metrics :obj:{`<cornac.metrics.RatingMetric>`, `<cornac.metrics.RankingMetric>`}, required
        A collection of metrics to use to evaluate the recommender models after reranking, \
        e.g., [Calibration, Recall].

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
    if len(reranking_metrics) == 0:
        return [], []

    avg_results = []
    user_results = [{} for _ in enumerate(reranking_metrics)]
    gt_mat = test_set.csr_matrix
    train_mat = train_set.csr_matrix
    val_mat = None if val_set is None else val_set.csr_matrix

    def pos_items(csr_row):
        return [
            item_idx
            for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
            if rating >= rating_threshold
        ]

    def get_gd_ratings(csr_row):
        return [rating
                for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
                ]

    def get_gd_idx(csr_row):
        return [item_idx
                for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
                ]
    not_evaluated_ids = {}
    not_evaluated_ids['total_users'] = len(test_set.user_indices)
    user_history_dict = OrderedDict()
    for user_idx in (test_set.user_indices):
        pos_item_idx = (
            []
            if train_set.is_unk_user(user_idx)
            else pos_items(train_mat.getrow(user_idx))
        )
        user_history_dict[user_idx] = pos_item_idx
    for user_idx in tqdm(
            test_set.user_indices, desc="Reranking evaluation", disable=not verbose, miniters=100
    ):
        test_pos_items = pos_items(gt_mat.getrow(
            user_idx))
        if len(test_pos_items) == 0:
            continue

        u_gt_pos = np.zeros(test_set.num_items, dtype='int')
        u_gt_pos[test_pos_items] = 1

        val_pos_items = [] if val_mat is None else pos_items(
            val_mat.getrow(user_idx))
        train_pos_items = (
            []
            if train_set.is_unk_user(user_idx)
            else pos_items(train_mat.getrow(user_idx))
        )

        u_gt_neg = np.ones(test_set.num_items, dtype='int')
        u_gt_neg[test_pos_items + val_pos_items + train_pos_items] = 0
        item_indices = None if exclude_unknowns else np.arange(
            test_set.num_items)
        pool_ids = np.arange(test_set.num_items)
        u_gt_rating = np.zeros(test_set.num_items)
        gd_item_idx = get_gd_idx(gt_mat.getrow(user_idx))
        gd_item_rating = get_gd_ratings(gt_mat.getrow(user_idx))
        for i in range(len(gd_item_idx)):
            # user ground truth rating
            u_gt_rating[gd_item_idx[i]] = gd_item_rating[i]
        # interacted and positive rating in training set
        user_history = user_history_dict[user_idx]
        # check if metrics contain Binomial
        globalProbs = []
        globalProbs_dict = {}
        for i, mt in enumerate(reranking_metrics):
            if "Binomial" in mt.name:
                global_prob = mt.globalFeatureProbs(user_history_dict)
                globalProbs.append(global_prob)
                globalProbs_dict["Binomial"] = global_prob
            else:
                globalProbs.append([])
        pd_other_users = []
        pd_other_users_dict = {}
        userIdx = list(test_set.user_indices)
        for i, mt in enumerate(reranking_metrics):
            if "Fragmentation" in mt.name:
                index = userIdx.index(user_idx)
                candidate_user_list = list(
                    userIdx[0:index]) + list(userIdx[index + 1:])
                samples = list(np.random.choice(
                    candidate_user_list, size=mt.n_samples, replace=False))
                if mt.k > 0:
                    sample_rank = [model.rank(x, item_indices)[
                        0][:mt.k] for x in samples]
                else:
                    sample_rank = [model.rank(x, item_indices)[0]
                                   for x in samples]
                pd_other_users.append(sample_rank)
                pd_other_users_dict["Fragmentation"] = sample_rank
            else:
                pd_other_users.append([])

        item_rank, item_scores = model.rank(user_idx, item_indices)
        reranking = RerankingAlgorithm(
            k=k,
            rerank=rerank,
            lambda_constant=lambda_constant,
            diversity_objective=diversity_objective,
            gt_ratings=u_gt_rating,
            u_gt_pos=u_gt_pos,
            u_gt_neg=u_gt_neg,
            rating_threshold=rating_threshold,
            globalProbs_dict=globalProbs_dict,
            user_history=user_history,
            pool_ids=pool_ids,
            pd_other_users_dict=pd_other_users_dict,
            user_idx=user_idx,
        )

        itemScoreDict = dict(zip(item_rank[:k], item_scores[item_rank[:k]]))
        item_rank_new = np.array(list(reranking.re_rank(itemScoreDict=itemScoreDict))
                                 + list(item_rank[k:]))
        item_scores_new = item_scores[item_rank_new]
        for i, mt in enumerate(reranking_metrics):
            mt_score = mt.compute(
                gt_pos=u_gt_pos,
                gt_neg=u_gt_neg,
                pd_rank=item_rank_new,
                pd_scores=item_scores_new,
                rating_threshold=rating_threshold,
                gt_ratings=u_gt_rating,
                globalProb=globalProbs[i],
                user_history=user_history,
                pool=pool_ids,
                pd_other_users=pd_other_users[i]
            )

            if mt_score is None:
                if mt.name not in not_evaluated_ids:
                    not_evaluated_ids[mt.name] = 1
                elif mt.name in not_evaluated_ids:
                    not_evaluated_ids[mt.name] = not_evaluated_ids[mt.name] + 1
            else:
                user_results[i][user_idx] = mt_score

        # avg results of diversity metrics
    for i, mt in enumerate(reranking_metrics):
        if len(user_results[i]) > 0:
            avg_results.append(
                sum(user_results[i].values()) / len(user_results[i]))
        else:
            avg_results.append(-1)

    rerank_parameter = {}
    rerank_parameter['rerank'] = rerank
    rerank_parameter['k'] = k
    rerank_parameter['lambda_constant'] = lambda_constant
    rerank_parameter['diversity_objective'] = diversity_objective[0].name.split(
        '@')[0]

    return avg_results, user_results, rerank_parameter


class BaseMethod:
    """Base Evaluation Method

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value)].

    rating_threshold: float, optional, default: 1.0
        Threshold used to binarize rating values into positive or negative feedback for
        model evaluation using ranking metrics (rating metrics are not affected).

    seed: int, optional, default: None
        Random seed for reproducibility.

    exclude_unknowns: bool, optional, default: True
        If `True`, unknown users and items will be ignored during model evaluation.

    verbose: bool, optional, default: False
        Output running log.

    """

    def __init__(
            self,
            data=None,
            fmt="UIR",
            rating_threshold=1.0,
            seed=None,
            exclude_unknowns=True,
            verbose=False,
            **kwargs
    ):
        self._data = data
        self.fmt = fmt
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.k = 0
        self.rerank = 0
        self.lambda_constant = 0
        self.diversity_objective = None
        self.reranking_metrics = None
        self.rating_threshold = rating_threshold
        self.exclude_unknowns = exclude_unknowns
        self.verbose = verbose
        self.seed = seed
        self.rng = get_rng(seed)
        self.global_uid_map = OrderedDict()
        self.global_iid_map = OrderedDict()

        self.user_feature = kwargs.get("user_feature", None)
        self.user_text = kwargs.get("user_text", None)
        self.user_image = kwargs.get("user_image", None)
        self.user_graph = kwargs.get("user_graph", None)
        self.item_feature = kwargs.get("item_feature", None)
        self.item_text = kwargs.get("item_text", None)
        self.item_image = kwargs.get("item_image", None)
        self.item_graph = kwargs.get("item_graph", None)
        self.sentiment = kwargs.get("sentiment", None)
        self.review_text = kwargs.get("review_text", None)

        if verbose:
            print("rating_threshold = {:.1f}".format(rating_threshold))
            print("exclude_unknowns = {}".format(exclude_unknowns))

    @property
    def total_users(self):
        return len(self.global_uid_map)

    @property
    def total_items(self):
        return len(self.global_iid_map)

    @property
    def user_feature(self):
        return self.__user_feature

    @property
    def user_text(self):
        return self.__user_text

    @user_feature.setter
    def user_feature(self, input_modality):
        if input_modality is not None and not isinstance(
                input_modality, FeatureModality
        ):
            raise ValueError(
                "input_modality has to be instance of FeatureModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_feature = input_modality

    @user_text.setter
    def user_text(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, TextModality):
            raise ValueError(
                "input_modality has to be instance of TextModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_text = input_modality

    @property
    def user_image(self):
        return self.__user_image

    @user_image.setter
    def user_image(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, ImageModality):
            raise ValueError(
                "input_modality has to be instance of ImageModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_image = input_modality

    @property
    def user_graph(self):
        return self.__user_graph

    @user_graph.setter
    def user_graph(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, GraphModality):
            raise ValueError(
                "input_modality has to be instance of GraphModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_graph = input_modality

    @property
    def item_feature(self):
        return self.__item_feature

    @property
    def item_text(self):
        return self.__item_text

    @item_feature.setter
    def item_feature(self, input_modality):
        if input_modality is not None and not isinstance(
                input_modality, FeatureModality
        ):
            raise ValueError(
                "input_modality has to be instance of FeatureModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_feature = input_modality

    @item_text.setter
    def item_text(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, TextModality):
            raise ValueError(
                "input_modality has to be instance of TextModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_text = input_modality

    @property
    def item_image(self):
        return self.__item_image

    @item_image.setter
    def item_image(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, ImageModality):
            raise ValueError(
                "input_modality has to be instance of ImageModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_image = input_modality

    @property
    def item_graph(self):
        return self.__item_graph

    @item_graph.setter
    def item_graph(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, GraphModality):
            raise ValueError(
                "input_modality has to be instance of GraphModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_graph = input_modality

    @property
    def sentiment(self):
        return self.__sentiment

    @sentiment.setter
    def sentiment(self, input_modality):
        if input_modality is not None and not isinstance(
                input_modality, SentimentModality
        ):
            raise ValueError(
                "input_modality has to be instance of SentimentModality but {}".format(
                    type(input_modality)
                )
            )
        self.__sentiment = input_modality

    @property
    def review_text(self):
        return self.__review_text

    @review_text.setter
    def review_text(self, input_modality):
        if input_modality is not None and not isinstance(
                input_modality, ReviewModality
        ):
            raise ValueError(
                "input_modality has to be instance of ReviewModality but {}".format(
                    type(input_modality)
                )
            )
        self.__review_text = input_modality

    def _reset(self):
        """Reset the random number generator for reproducibility"""
        self.rng = get_rng(self.seed)
        self.test_set = self.test_set.reset()

    def _organize_metrics(self, metrics):
        """Organize metrics according to their types (rating or raking)

        Parameters
        ----------
        metrics: :obj:`iterable`
            List of metrics.

        """
        if isinstance(metrics, dict):
            self.rating_metrics = metrics.get("rating", [])
            self.ranking_metrics = metrics.get("ranking", [])
            self.diversity_metrics = metrics.get("diversity", [])

        elif isinstance(metrics, list):
            self.rating_metrics = []
            self.ranking_metrics = []
            self.diversity_metrics = []
            for mt in metrics:
                if isinstance(mt, RatingMetric):
                    self.rating_metrics.append(mt)
                elif isinstance(mt, RankingMetric) and hasattr(mt.k, "__len__"):
                    self.ranking_metrics.extend(
                        [mt.__class__(k=_k) for _k in sorted(set(mt.k))]
                    )
                elif isinstance(mt, DiversityMetric):
                    self.diversity_metrics.append(mt)
                else:
                    self.ranking_metrics.append(mt)
        else:
            raise ValueError("Type of metrics has to be either dict or list!")

        # sort metrics by name
        self.rating_metrics = sorted(
            self.rating_metrics, key=lambda mt: mt.name)
        self.ranking_metrics = sorted(
            self.ranking_metrics, key=lambda mt: mt.name)
        self.diversity_metrics = sorted(
            self.diversity_metrics, key=lambda mt: mt.name)

    def _organize_reranking_metrics(self, metrics):
        if not isinstance(metrics, list):
            raise ValueError("Type of metrics has to be list!")
        ranking_metrics = []
        diversity_metrics = []
        for mt in metrics:
            if isinstance(mt, RankingMetric) and hasattr(mt.k, "__len__"):
                ranking_metrics.extend(
                    [mt.__class__(k=_k) for _k in sorted(set(mt.k))]
                )
            elif isinstance(mt, DiversityMetric):
                diversity_metrics.append(mt)
            else:
                ranking_metrics.append(mt)
        # sort metrics by name
        ranking_metrics = sorted(
            ranking_metrics, key=lambda mt: mt.name)
        diversity_metrics = sorted(
            diversity_metrics, key=lambda mt: mt.name)
        metrics = ranking_metrics + diversity_metrics

        return metrics

    def _build_datasets(self, train_data, test_data, val_data=None):
        self.train_set = Dataset.build(
            data=train_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            seed=self.seed,
            exclude_unknowns=False,
        )
        if self.verbose:
            print("---")
            print("Training data:")
            print("Number of users = {}".format(self.train_set.num_users))
            print("Number of items = {}".format(self.train_set.num_items))
            print("Number of ratings = {}".format(self.train_set.num_ratings))
            print("Max rating = {:.1f}".format(self.train_set.max_rating))
            print("Min rating = {:.1f}".format(self.train_set.min_rating))
            print("Global mean = {:.1f}".format(self.train_set.global_mean))

        self.test_set = Dataset.build(
            data=test_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            seed=self.seed,
            exclude_unknowns=self.exclude_unknowns,
        )
        if self.verbose:
            print("---")
            print("Test data:")
            print("Number of users = {}".format(len(self.test_set.uid_map)))
            print("Number of items = {}".format(len(self.test_set.iid_map)))
            print("Number of ratings = {}".format(self.test_set.num_ratings))
            print(
                "Number of unknown users = {}".format(
                    self.test_set.num_users - self.train_set.num_users
                )
            )
            print(
                "Number of unknown items = {}".format(
                    self.test_set.num_items - self.train_set.num_items
                )
            )

        if val_data is not None and len(val_data) > 0:
            self.val_set = Dataset.build(
                data=val_data,
                fmt=self.fmt,
                global_uid_map=self.global_uid_map,
                global_iid_map=self.global_iid_map,
                seed=self.seed,
                exclude_unknowns=self.exclude_unknowns,
            )
            if self.verbose:
                print("---")
                print("Validation data:")
                print("Number of users = {}".format(len(self.val_set.uid_map)))
                print("Number of items = {}".format(len(self.val_set.iid_map)))
                print("Number of ratings = {}".format(self.val_set.num_ratings))

        if self.verbose:
            print("---")
            print("Total users = {}".format(self.total_users))
            print("Total items = {}".format(self.total_items))

        self.train_set.total_users = self.total_users
        self.train_set.total_items = self.total_items

    def _build_modalities(self):
        for user_modality in [
            self.user_feature,
            self.user_text,
            self.user_image,
            self.user_graph,
        ]:
            if user_modality is None:
                continue
            user_modality.build(
                id_map=self.global_uid_map,
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for item_modality in [
            self.item_feature,
            self.item_text,
            self.item_image,
            self.item_graph,
        ]:
            if item_modality is None:
                continue
            item_modality.build(
                id_map=self.global_iid_map,
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for modality in [self.sentiment, self.review_text]:
            if modality is None:
                continue
            modality.build(
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        self.add_modalities(
            user_feature=self.user_feature,
            user_text=self.user_text,
            user_image=self.user_image,
            user_graph=self.user_graph,
            item_feature=self.item_feature,
            item_text=self.item_text,
            item_image=self.item_image,
            item_graph=self.item_graph,
            sentiment=self.sentiment,
            review_text=self.review_text,
        )

    def add_modalities(self, **kwargs):
        """
        Add successfully built modalities to all datasets. This is handy for
        seperately built modalities that are not invoked in the build method.
        """
        self.user_feature = kwargs.get("user_feature", None)
        self.user_text = kwargs.get("user_text", None)
        self.user_image = kwargs.get("user_image", None)
        self.user_graph = kwargs.get("user_graph", None)
        self.item_feature = kwargs.get("item_feature", None)
        self.item_text = kwargs.get("item_text", None)
        self.item_image = kwargs.get("item_image", None)
        self.item_graph = kwargs.get("item_graph", None)
        self.sentiment = kwargs.get("sentiment", None)
        self.review_text = kwargs.get("review_text", None)

        for data_set in [self.train_set, self.test_set, self.val_set]:
            if data_set is None:
                continue
            data_set.add_modalities(
                user_feature=self.user_feature,
                user_text=self.user_text,
                user_image=self.user_image,
                user_graph=self.user_graph,
                item_feature=self.item_feature,
                item_text=self.item_text,
                item_image=self.item_image,
                item_graph=self.item_graph,
                sentiment=self.sentiment,
                review_text=self.review_text,
            )

    def build(self, train_data, test_data, val_data=None):
        if train_data is None or len(train_data) == 0:
            raise ValueError("train_data is required but None or empty!")
        if test_data is None or len(test_data) == 0:
            raise ValueError("test_data is required but None or empty!")

        self.global_uid_map.clear()
        self.global_iid_map.clear()

        self._build_datasets(train_data, test_data, val_data)
        self._build_modalities()

        return self

    def _eval(self, model, test_set, val_set, user_based):
        metric_avg_results = OrderedDict()
        metric_user_results = OrderedDict()
        user_info = OrderedDict()
        model_parameter = OrderedDict()
        avg_results, user_results = rating_eval(
            model=model,
            metrics=self.rating_metrics,
            test_set=test_set,
            user_based=user_based,
            verbose=self.verbose,
        )
        for i, mt in enumerate(self.rating_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        avg_results, user_results = ranking_eval(
            model=model,
            metrics=self.ranking_metrics,
            train_set=self.train_set,
            test_set=test_set,
            val_set=val_set,
            rating_threshold=self.rating_threshold,
            exclude_unknowns=self.exclude_unknowns,
            verbose=self.verbose,
        )
        for i, mt in enumerate(self.ranking_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        # diversity evaluate
        avg_results, user_results, user_info_result = diversity_eval(
            model=model,
            metrics=self.diversity_metrics,
            train_set=self.train_set,
            test_set=test_set,
            val_set=val_set,
            rating_threshold=self.rating_threshold,
            exclude_unknowns=self.exclude_unknowns,
            verbose=self.verbose,
        )
        for i, mt in enumerate(self.diversity_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]
            user_info[mt.name] = user_info_result[i]
        if len(user_info_result) > 0:
            user_info['total_user_number'] = user_info_result[-1]
        return Result(model.name, metric_avg_results, metric_user_results,
                      user_info=user_info, model_parameter=model_parameter, rerank_parameter=None)

    def _eval_rerank(self, model, test_set, val_set, k, rerank, lambda_constant, diversity_objective, reranking_metrics):
        metric_avg_results = OrderedDict()
        metric_user_results = OrderedDict()
        rerank_parameter = OrderedDict()
        avg_results, user_results, rerank_parameter = reranking_diversity_eval(
            model=model,
            train_set=self.train_set,
            test_set=test_set,
            val_set=val_set,
            k=k,
            rerank=rerank,
            lambda_constant=lambda_constant,
            diversity_objective=diversity_objective,
            reranking_metrics=reranking_metrics,
            rating_threshold=self.rating_threshold,
            exclude_unknowns=self.exclude_unknowns,
            verbose=self.verbose,
        )
        for i, mt in enumerate(reranking_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]
        return Result(model.name, metric_avg_results, metric_user_results,
                      user_info=None, model_parameter=None, rerank_parameter=rerank_parameter)

    def evaluate(self, model, metrics, user_based, show_validation=True):
        """Evaluate given models according to given metrics

        Parameters
        ----------
        model: :obj:`cornac.models.Recommender`
            Recommender model to be evaluated.

        metrics: :obj:`iterable`
            List of metrics.

        user_based: bool, required
            Evaluation strategy for the rating metrics. Whether results
            are averaging based on number of users or number of ratings.

        show_validation: bool, optional, default: True
            Whether to show the results on validation set (if exists).

        Returns
        -------
        res: :obj:`cornac.experiment.Result`
        """
        if self.train_set is None:
            raise ValueError("train_set is required but None!")
        if self.test_set is None:
            raise ValueError("test_set is required but None!")

        self._reset()
        self._organize_metrics(metrics)

        ###########
        # FITTING #
        ###########
        if self.verbose:
            print("\n[{}] Training started!".format(model.name))

        start = time.time()
        model.fit(self.train_set, self.val_set)
        train_time = time.time() - start

        ##############
        # EVALUATION #
        ##############
        if self.verbose:
            print("\n[{}] Evaluation started!".format(model.name))

        start = time.time()
        test_result = self._eval(
            model=model,
            test_set=self.test_set,
            val_set=self.val_set,
            user_based=user_based,
        )
        test_time = time.time() - start
        test_result.metric_avg_results["Train (s)"] = train_time
        test_result.metric_avg_results["Test (s)"] = test_time
        # read parameters that need to save.
        all_para = save_model_parameter()
        parameter_values = {}
        # use attributes of objects to check parameter information.
        if model.name in all_para:
            para = all_para[model.name]
            for att in dir(model):
                if att in para:
                    res = getattr(model, att)
                    if isinstance(res, list):
                        res = ','.join(str(e) for e in res)
                    parameter_values[att] = res
        test_result.model_parameter = parameter_values

        val_result = None
        if show_validation and self.val_set is not None:
            start = time.time()
            val_result = self._eval(
                model=model, test_set=self.val_set, val_set=None, user_based=user_based
            )
            val_time = time.time() - start
            val_result.metric_avg_results["Time (s)"] = val_time
        return test_result, val_result

    def evaluate_rerank(self, model, metrics, k, rerank, lambda_constant, diversity_objective,
                        reranking_metrics, show_validation=True):
        """Evaluate given models according to given metrics after re-ranking

        Parameters
        ----------
        model: :obj:`cornac.models.Recommender`
            Recommender model to be evaluated.

        metrics: :obj:`iterable`
            List of metrics.

        k: int
            This parameter is only useful if you are considering re-ranking.
            e.g., 20, the top 20 items recommended will be reranked.

        rerank: int
            This parameter is only useful if you are considering re-ranking.
            e.g., 200, the top @k items recommended to be reranked will be chosen from top 200 items.

        lambda_constant: float
            weight factor of the diversity metrics.

        diversity_objective :obj:{`<cornac.metrics.DiversityMetric>`}, required
            A collection of metrics to use to evaluate the recommender models for reranking, \
            e.g., [Calibration].

        reranking_metrics :obj:{`<cornac.metrics.RatingMetric>`, `<cornac.metrics.RankingMetric>`}, required
            A collection of metrics to use to evaluate the recommender models after reranking, \
            e.g., [Calibration, Recall].

        show_validation: bool, optional, default: True
            Whether to show the results on validation set (if exists).

        Returns
        -------
        res: :obj:`cornac.experiment.Result`
        """
        if self.train_set is None:
            raise ValueError("train_set is required but None!")
        if self.test_set is None:
            raise ValueError("test_set is required but None!")

        self._reset()
        self._organize_metrics(metrics)

        ##############
        # EVALUATION #
        ##############
        if self.verbose:
            print("\n[{}] Re-ranking started!".format(model.name))

        self.k = k
        self.rerank = rerank
        self.lambda_constant = lambda_constant
        self.diversity_objective = diversity_objective
        self.reranking_metrics = self._organize_reranking_metrics(
            reranking_metrics)

        start = time.time()
        test_result = self._eval_rerank(
            model=model,
            test_set=self.test_set,
            val_set=self.val_set,
            k=self.k,
            rerank=self.rerank,
            lambda_constant=self.lambda_constant,
            diversity_objective=self.diversity_objective,
            reranking_metrics=self.reranking_metrics,
        )
        test_time = time.time() - start
        test_result.metric_avg_results["Re-rank Time (s)"] = test_time
        val_result = None
        if show_validation and self.val_set is not None:
            start = time.time()
            val_result = self._eval_rerank(
                model=model, test_set=self.val_set, val_set=None, k=self.k, rerank=self.rerank,
                lambda_constant=self.lambda_constant, diversity_objective=self.diversity_objective,
                reranking_metrics=self.reranking_metrics,
            )
            val_time = time.time() - start
            val_result.metric_avg_results["Time (s)"] = val_time

        return test_result, val_result

    @classmethod
    def from_splits(
            cls,
            train_data,
            test_data,
            val_data=None,
            fmt="UIR",
            rating_threshold=1.0,
            exclude_unknowns=False,
            seed=None,
            verbose=False,
            **kwargs
    ):
        """Constructing evaluation method given data.

        Parameters
        ----------
        train_data: array-like
            Training data

        test_data: array-like
            Test data

        val_data: array-like, optional, default: None
            Validation data

        fmt: str, default: 'UIR'
            Format of the input data. Currently, we are supporting:

            'UIR': User, Item, Rating
            'UIRT': User, Item, Rating, Timestamp

        rating_threshold: float, default: 1.0
            Threshold to decide positive or negative preferences.

        exclude_unknowns: bool, default: False
            Whether to exclude unknown users/items in evaluation.

        seed: int, optional, default: None
            Random seed for reproduce the splitting.

        verbose: bool, default: False
            The verbosity flag.

        Returns
        -------
        method: :obj:`<cornac.eval_methods.BaseMethod>`
            Evaluation method object.

        """
        method = cls(
            fmt=fmt,
            rating_threshold=rating_threshold,
            exclude_unknowns=exclude_unknowns,
            seed=seed,
            verbose=verbose,
            **kwargs
        )

        return method.build(
            train_data=train_data, test_data=test_data, val_data=val_data
        )
