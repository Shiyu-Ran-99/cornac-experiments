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

import os
from datetime import datetime

from .result import ExperimentResult
from .result import CVExperimentResult
from ..metrics.rating import RatingMetric
from ..metrics.ranking import RankingMetric
from ..metrics.diversity import DiversityMetric
from ..models.recommender import Recommender
from collections import OrderedDict
import json


class Experiment:
    """ Experiment Class

    Parameters
    ----------
    eval_method: :obj:`<cornac.eval_methods.BaseMethod>`, required
        The evaluation method (e.g., RatioSplit).

    models: array of :obj:`<cornac.models.Recommender>`, required
        A collection of recommender models to evaluate, e.g., [C2PF, HPF, PMF].

    metrics: array of :obj:{`<cornac.metrics.RatingMetric>`, `<cornac.metrics.RankingMetric>`}, required
        A collection of metrics to use to evaluate the recommender models, \
        e.g., [NDCG, MRR, Recall].

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

    user_based: bool, optional, default: True
        This parameter is only useful if you are considering rating metrics. When True, first the average performance \
        for every user is computed, then the obtained values are averaged to return the final result.
        If `False`, results will be averaged over the number of ratings.

    show_validation: bool, optional, default: True 
        Whether to show the results on validation set (if exists).

    save_dir: str, optional, default: None
        Path to a directory for storing trained models and logs. If None, 
        models will NOT be stored and logs will be saved in the current working directory.

    Attributes
    ----------
    result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment 
        on the test set, initially it is set to None.

    val_result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment
        on the validation set (if exists), initially it is set to None.

    rerank_result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment
        on the test set (if exists) after re-ranking, initially it is set to None.

    """

    def __init__(
        self,
        eval_method,
        models,
        metrics,
        k=0,
        rerank=0,
        lambda_constant=0,
        diversity_objective=None,
        reranking_metrics=None,
        user_based=True,
        show_validation=True,
        verbose=False,
        save_dir=None,
    ):
        self.eval_method = eval_method
        self.models = self._validate_models(models)
        self.metrics = self._validate_metrics(metrics)
        self.k = k
        self.rerank = rerank
        self.lambda_constant = lambda_constant
        self.diversity_objective = self._validate_reranking_metrics(diversity_objective)
        self.reranking_metrics = self._validate_reranking_metrics(reranking_metrics)
        self.user_based = user_based
        self.show_validation = show_validation
        self.verbose = verbose
        self.save_dir = save_dir
        self.result = None
        self.val_result = None
        self.rerank_result = None
        self.val_result_rerank = None

    @staticmethod
    def _validate_models(input_models):
        if not hasattr(input_models, "__len__"):
            raise ValueError(
                "models have to be an array but {}".format(type(input_models))
            )

        valid_models = []
        for model in input_models:
            if isinstance(model, Recommender):
                valid_models.append(model)
        return valid_models

    @staticmethod
    def _validate_metrics(input_metrics):
        if input_metrics == None:
            return None
        if not hasattr(input_metrics, "__len__"):
            raise ValueError(
                "metrics have to be an array but {}".format(
                    type(input_metrics))
            )

        valid_metrics = []
        for metric in input_metrics:
            if isinstance(metric, RatingMetric) or isinstance(metric, RankingMetric) or isinstance(metric, DiversityMetric):
                valid_metrics.append(metric)
        return valid_metrics

    @staticmethod
    def _validate_reranking_metrics(input_metrics):
        if input_metrics == None:
            return None
        if not hasattr(input_metrics, "__len__"):
            raise ValueError(
                "metrics have to be an array but {}".format(
                    type(input_metrics))
            )

        valid_metrics = []
        for metric in input_metrics:
            if isinstance(metric, RankingMetric) or isinstance(metric, DiversityMetric):
                valid_metrics.append(metric)
        return valid_metrics

    def _create_result(self):
        from ..eval_methods.cross_validation import CrossValidation
        from ..eval_methods.propensity_stratified_evaluation import (
            PropensityStratifiedEvaluation,
        )

        if isinstance(self.eval_method, CrossValidation) or isinstance(
            self.eval_method, PropensityStratifiedEvaluation
        ):
            self.result = CVExperimentResult()
        else:
            self.result = ExperimentResult()
            if self.show_validation and self.eval_method.val_set is not None:
                self.val_result = ExperimentResult()
            if self.k != 0:
                self.rerank_result = ExperimentResult()
                if self.show_validation and self.eval_method.val_set is not None:
                    self.val_result_rerank = ExperimentResult()

    def run(self):
        """Run the Cornac experiment"""
        self._create_result()

        for model in self.models:
            test_result, val_result = self.eval_method.evaluate(
                model=model,
                metrics=self.metrics,
                user_based=self.user_based,
                show_validation=self.show_validation,
            )

            self.result.append(test_result)
            if self.val_result is not None:
                self.val_result.append(val_result)

            if self.rerank == 0 or self.k == 0 or self.k <= self.rerank or \
                    self.diversity_objective is None or len(self.diversity_objective) == 0 or \
                self.reranking_metrics is None or len(self.reranking_metrics) == 0:
                print("If you want to re-rank the results, please input rerank < k and \
                    both rerank and k are larger than 0, also input the diversity metrics that you want to improve \
                    and the metrics that you want to compute after reranking in a proper way!")
            else:
                test_result_rerank, val_result_rerank = self.eval_method.evaluate_rerank(
                    model=model,
                    metrics=self.metrics,
                    k=self.k,
                    rerank=self.rerank,
                    lambda_constant=self.lambda_constant,
                    diversity_objective=self.diversity_objective,
                    reranking_metrics=self.reranking_metrics,
                    show_validation=self.show_validation,
                )

                self.rerank_result.append(test_result_rerank)
                if self.val_result_rerank is not None:
                    self.val_result_rerank.append(val_result_rerank)

            if not isinstance(self.result, CVExperimentResult):
                model.save(self.save_dir)

        output = ""
        if self.val_result is not None:
            output += "\nVALIDATION:\n...\n{}".format(self.val_result)
        output += "\nTEST:\n...\n{}".format(self.result)

        if self.rerank != 0 and self.k != 0 and self.k > self.rerank:
            if self.val_result_rerank is not None:
                output += "\nVALIDATION-RE-RANKING:\n...\n{}".format(
                    self.val_result_rerank)
            output += "\nRE-RANKING:\n...\n{}".format(self.rerank_result)

        print(output)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        save_dir = "." if self.save_dir is None else self.save_dir
        output_file = os.path.join(
            save_dir, "CornacExp-{}.log".format(timestamp))
        with open(output_file, "w") as f:
            f.write(output)

        result0 = OrderedDict()
        if isinstance(self.result, CVExperimentResult):
            pass
        else:
            if self.result is not None and len(self.result) != 0:
                for i in self.result:
                    result1 = OrderedDict()
                    if bool(i.metric_avg_results):
                        result1['metric_avg_results'] = i.metric_avg_results
                    if bool(i.user_info):
                        result1['user_info'] = i.user_info
                    if bool(i.model_parameter):
                        result1['model_parameter'] = i.model_parameter
                    result0[i.model_name] = result1
            if self.rerank_result is not None and len(self.rerank_result) != 0:
                for i in self.rerank_result:
                    result2 = OrderedDict()
                    if bool(i.metric_avg_results):
                        result2 = i.metric_avg_results
                    if bool(i.rerank_parameter):
                        result2['rerank_parameter'] = i.rerank_parameter
                    result0[i.model_name]['Re-Rank'] = result2

        if len(result0) != 0:
            output_json = os.path.join(
                save_dir, "CornacExp-{}.json".format(timestamp))
            with open(output_json, 'w') as fout:
                json.dump(result0, fout)
