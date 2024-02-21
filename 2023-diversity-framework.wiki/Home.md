# Project Wiki: Diversity Framework for Recommender System

The wiki page for this project includes a general explanation of the project, as well as detailed implementation descriptions for each of the three parts. The page also includes information on how the Cornac framework was extended to evaluate various recommender systems in terms of diversity. Additionally, the page also includes code snippets and examples to help users get started with the framework.

The goal of this master project is to extend the existing recommender system evaluation framework, [Cornac](https://cornac.readthedocs.io/en/latest/), to go beyond accuracy and include diversity as an important evaluation metric. Overall, the project consists of three main parts: 1). enhancing datasets to be able to deal with diversity, 2). implementing diversity metrics, and 3). extending Cornac's recommendation algorithms.

Overall, this project represents an important step forward in the development of recommender systems that take diversity into account. By enhancing existing datasets, developing new diversity metrics, and extending Cornac's recommendation algorithm, we aim to create a more inclusive and diverse online ecosystem. The resulting framework will be an invaluable tool for researchers and practitioners working in the field of recommender systems.

---

## Dataset Enhancement

The first part of the project consists of enhancing existing datasets to include diversity-related features such as item category, movie genre, and political views of news articles. This allows for more accurate evaluation of diversity metrics and provides a more nuanced understanding of the impact of diversity on recommendation performance. We built the enhancement pipelines to extend 1) Cornac built-in datasets and 2) custom datasets from users (i.e., datasets that are currently not included in the Cornac framework).

- [Cornac built-in dataset enhancement pipeline](Cornac-Data-Enhancement)
- [Custom dataset enhancement pipeline](Custom-Data-Enhancement)

The enhanced dataset provides several key benefits. Firstly, by incorporating extra features, the dataset becomes richer and more diverse, allowing recommendation algorithms to consider a wider range of item characteristics, thereby improving diversity. Secondly, the availability of additional features empowers recommendation algorithms to make more informed and diverse recommendations, resulting in an overall enhancement of recommendation quality. Lastly, the enhanced dataset offers users greater customizability, enabling them to experiment with various feature combinations and algorithms to tailor the recommendation system to their specific needs.

---
## Data Loader
The module's primary function is to handle the process of bringing external data into the cornac framework for processing and analysis. Users can follow the [Custom dataset enhancement pipeline](Custom-Data-Enhancement) to generate a json file which contains nested dictionary with its id and extended features pair.  
Then, it's recommended to divide this JSON file into separate feature files, such as "category.json" and "sentiment.json". Through specialized functions, these feature JSON files can be loaded and transformed into the necessary format required by the diversity framework. Detailed information about the data loader functions is provided on the [Mind-Data-Loader](Mind-Data-Loader) page.


## Diversity Metrics

The second part of the project is implementing diversity metrics that can be used to evaluate the performance of recommender systems. The diversity metrics implemented by us are listed below.

| Metric                        | Feature Type                               | Additional Information          | Documentation                         |
| ----------------------------- | ------------------------------------------ | ------------------------------- | ------------------------------------- |
| calibration                   | genre/complexity                           | user history                    | [Link](Calibration)                   |
| fragmentation                 | story                                      | other users' recommendations    | [Link](Fragmentation)                 |
| activation                    | sentiment                                  | pool of available items         | [Link](Activation)                    |
| representation                | entities                                   | pool of available items         | [Link](Representation)                |
| alternative voices            | minority score and majority score | pool of available items         | [Link](Alternative-voices)            |
| Gini coefficient              | genre                                      | \                               | [Link](Gini-coefficient)              |
| $\alpha$-nDCG                 |            genre                     | user history | [Link](Alpha-NDCG)                    |
| intra-list diversity          | feature vectors                            | \                               | [Link](Intra-list-diversity)          |
| expected intra-list diversity | feature vectors                            | ground-truth relevance          | [Link](Expected-Intra-list-diversity) |
| binomial diversity            | genre                                      | user history                    | [Link](Binomial)                      |

---


## Recommender Algorithms

The final part of the project is extending Cornac's recommendation algorithm to include diversity as a key factor in the recommendation process. The new algorithm will be designed to optimize both accuracy and diversity, taking into account the different diversity metrics developed in the previous step. The new algorithm will be evaluated on a range of datasets to demonstrate its effectiveness in improving diversity in recommender systems.

### Cornac Algorithms

Testing existing algorithms with diversity functions. This task assumes a good understanding of the algorithms that are currently available in Cornac. We mainly test the algorithms which are suitable for text modality data, including 6 algorithms: ctr, hft, conv_mf, cdl, cdr and cvae. Also, we experiment with new dataset: MIND, researching whether this dataset is available for cornac algorithms.

- [Collaborative Topic Regression (CTR)](<Collaborative-Topic-Regression-(CTR)>)
- [Hidden Factors and Hidden Topics (HFT)](<Hidden-Factors-and-Hidden-Topics-(HFT)>)
- [Convolutional Matrix Factorization (ConvMF)](<Convolutional-Matrix-Factorization-(ConvMF)>)
- [Collaborative Deep Learning (CDL)](<Collaborative-Deep-Learning-(CDL)>)
- [Collarborative Deep Ranking (CDR)](<Collaborative-Deep-Ranking-(CDR)>)
- [Collaborative Variational Autoencoder (CVAE)](<Collaborative-Variational-Autoencoder-(CVAE)>)

### Integrate Three New Algorithms into Cornac

- [Neural Matrix Factorization without Sampling (ENMF)](<Neural-Matrix-Factorization-without-Sampling(ENMF-algorithm)>)
- [Multinomial Denoising Autoencoder (DAE)](<Multinomial-Denoising-Autoencoder(DAE-algorithm)>)
- [Neural News Recommendation with Long- and Short-term User Representations(LSTUR)](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/wikis/Neural-News-Recommendation-with-Long-and-Short-term-User-Representations(LSTUR))

### Optimize Diversity Metrics

- [Re-rank Recommendations](Re-rank)
- [Select Top K Diversed Recommended Items](Select-Top-5-Diversed-Items)
- [Re-rank TOP K from a set of Candidates](Re-rank-TOP-K-from-a-set-of-Candidates)

### Benchmark Experiments
- [Benchmark Methods](Benchmark-Methods)

## New Cornac Framework with Diversity Metrics and Algorithms
### Gettting Started: your first Cornac diversity experiment
![flow](uploads/c55e1557c8c1f0e27090c5106f9ab8a2/flow.jpg)
<div align=center><i>Flow of an Experiment in Diversity Cornac</i></div>

```python
import cornac
from cornac.eval_methods import RatioSplit
from cornac.metrics import Recall, AUC
from cornac.datasets import mind as mind
from cornac.metrics import GiniCoeff
from cornac.metrics import Calibration
from cornac.metrics import AlternativeVoices
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer

# load the MIND dataset and features, split the data based on ratio
feedback = mind.load_feedback(fpath="./tests/enriched_data/mind_uir_20k.csv")
category = mind.load_category(fpath="./tests/enriched_data/category.json")
genre = mind.load_category_multi(fpath="./tests/enriched_data/category.json")
min_maj = mind.load_min_maj(fpath="./tests/enriched_data/min_maj.json")

text_dict = mind.load_text(fpath="./tests/enriched_data/text.json")
text = list(text_dict.values())
item_ids = list(text_dict.keys())
item_text_modality = TextModality(corpus=text,ids=item_ids,tokenizer=BaseTokenizer(sep=" ", stop_words="english"),max_vocab=8000,max_doc_freq=0.5,)
mind_ratio_split = RatioSplit(data=feedback,test_size=0.2,exclude_unknowns=True,item_text=item_text_modality,verbose=True,seed=123,rating_threshold=0.5)

# build features
Item_category = mind.build(data=category, id_map=mind_ratio_split.train_set.iid_map)
Item_min_major = mind.build(data=min_maj, id_map=mind_ratio_split.train_set.iid_map)
Item_genre = mind.build(data=genre, id_map=mind_ratio_split.train_set.iid_map)

# define metrics to evaluate the models
metrics = [AUC(), Recall(k=10), Calibration(item_feature=Item_category,data_type="category", divergence_type='JS', k=10),GiniCoeff(item_genre=Item_genre, k=10),AlternativeVoices(item_minor_major=Item_min_major, divergence_type='JS', k=10)]
# define re-ranking metrics
reranking_metrics = [AUC(), Recall(k=10), Calibration(item_feature=Item_category,data_type="category", divergence_type='JS', k=10),GiniCoeff(item_genre=Item_genre, k=10),AlternativeVoices(item_minor_major=Item_min_major, divergence_type='JS', k=10)]

# initialize models
models = [cornac.models.CTR(k=10,lambda_u=1000, lambda_v=10,max_iter=50)]

# put it together in an experiment
cornac.Experiment(
        eval_method=mind_ratio_split,
        models=models,
        metrics=metrics,
        k=50,  # the number of candidate set
        rerank=10,  # the number of re-ranking items
        lambda_constant=0,
        diversity_objective=[Calibration(item_feature=Item_category,data_type="category",divergence_type='JS')],
        reranking_metrics=reranking_metrics
).run()
```
**Output:**

TEST:

|...|  AUC | Recall@10 | AltVoices_mainstream@10 | Calibration_category@10 | GiniCoeff@10 | Train (s) | Test (s)|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| CTR | 0.9170 |    0.0432 |0.1149 |0.6233 | 0.7751 | 11113.0249 | 12746.5376 |

Number of Users in Diversity Metric Evaluation

|...| AltVoices_mainstream@10 | Calibration_category@10 | GiniCoeff@10 | total_user_number
|:----:|:----:|:----:|:----:|:----:|
CTR |17739 | 17670 | 17739 | 19663 |

Hyper-parameters

|...|  k | lambda_u | lambda_v |
|:----:|:----:|:----:|:----:|
CTR | 400 |   1000.0 |     10.0 |

RE-RANKING:

|...| Recall@10 | AltVoices_mainstream@10 | Calibration_category@10 | GiniCoeff@10 |Re-rank Time (s)|
|:----:|:----:|:----:|:----:|:----:|:----:|
|CTR | 0.0263 | 0.1240 | 0.9652 |0.8657 | 4912.8346|

Rerank-parameters

|...| rerank |  k | lambda_constant |  diversity_objective |
|:----:|:----:|:----:|:----:|:----:|
|CTR |10 | 50 | 0 | Calibration_category|

This experiment will also output JSON file, so it's more convenient for users to analyze the results. For more details, please take a look at this [script](https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-diversity-framework/-/blob/merge_reranking/main.py?ref_type=heads), it includes all experiment details and how we do the benchmarks.