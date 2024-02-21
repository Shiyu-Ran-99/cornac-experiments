## Goal of the metric

The activation metric is part of the metrics incorporated in the RADio paper, as detailed in the research conducted by Vrijenhoek et al.(2022). The Activation score assigned to the recommended items for each user indicates whether the articles lean towards being activating or neutral in nature.

The sentiment value is continuous, ranging between -1 and 1. Many sentiment analysis methods assess a text and provide a score ranging from 0 to 1 for positive emotions, -1 to 0 for negative sentiments, and 0 for complete neutrality. The activation metric utilizes the absolute sentiment score of an article to estimate the height of the emotion conveyed, thereby determining the level of Activation. The recommendations are compared to the available pool of data.

The output activation value of 0 indicates an identical distribution between the recommendations and the pool, implying that the recommender system presents content with the same level of activation as what was available in the data pool. As the value increases, the level of divergence also increases.

## Description

The computation of Activation requires item sentiments. The item sentiments should be provided as a dictionary using the format {item idx: item sentiments}, where the item idx corresponds to the internal item id used by the cornac system, rather than the original item indices. Item sentiments are continous values, ranging between -1 and 1. Our metric calculation will transform these sentiments into absolute values to meet the requirements specified in the subsequent formula.

### Parameters

**item_sentiment**: Dictionary. {item idx: item sentiments}. Contains item indices mapped to their respective sentiments.

**divergence_type**: (Optional) string that determines the divergence method employed. The choices are "JS" or "KL". The default value is "KL." For further detail, please refer to [Link](Divergence).

**discount**: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness. If you want to consider position in the recommendation, set it to True.

**n_bins**: (Optional) Integer. Determines the number of bins used for discretizing continuous data into intervals. The default value is 5.

**k**: (Optional) Integer. Rank cutoff @N. The default value is -1. When using default value -1, all items in the recommendation lists are evaluated.

## Formula

$Activation=Act(P(k|S),  Q^{\*}(k|R)) = \sum\limits_{k} Q^{\*}(k|R)f(\frac{P(k|S)}{Q^{\*}(k|R)})$  
The elements in the distribution Q represent the absolute sentiments of the recommended items.
The context P refers to the absolute sentiment of the overall supply of available items in the data pool.  
$P(k|S)$ represents the distribution of binned Activation scores (k) among the pool of available items (S)at a specific point. $Q^{\*}(k|R)$ expresses the same, but for the binned Activation scores in the recommendation distribution, taking into account the ranking awareness.

## Reference

Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September). RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).
