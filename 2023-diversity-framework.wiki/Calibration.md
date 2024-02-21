## Goal of the metric

The Calibration metric measures the extent to which the issued recommendations align with the user's preferences. A score of 0 represents perfect Calibration, indicating a precise match/alignment between the recommendations and the user's preferences. A higher score indicates a greater deviation from the user's preferences.

Calibration has two aspects to look at. The first aspect is evaluating the divergence in the distributions of categorical information, such as news topics or movie genres, between the current recommendations provided to the user and the user history. The second aspect of calibration is to accommodate for the user's preferences in terms of article complexity. This allows the reader to receive content that aligns with their information needs and article complexity. For instance, a user might possess expertise in politics but less so in medicine, leading them to prefer more complex articles in the former and relatively simpler ones in the latter.

## Description

To calculate the Calibration metric, the item features should be provided in the format of {item idx: item feature}. The data type, specified as either "category" or "complexity," must also be indicated.

When the data type is set as "category," the item feature is expected to consist of a single categorical value for each item. On the other hand, when the data type is specified as "complexity," the item feature should be a continuous value for each item. For instance, article complexity can be determined using the Flesch-Kincaid reading ease test as a measure.

## Parameters

**item_feature**: A dictionary that maps item indices to their respective categories or complexities. Categories are discrete values, where each item has a single categorical value selected from options such as {"action", "comedy", ...}. For example, {1: "action", 2:"comedy",... }. On the other hand, complexities are continuous values. For example, {1: 17, 2 : 22,... }.

**data_type**: A string indicating the type of data, either "category" or "complexity."

**divergence_type**: (Optional) A string determining the employed method of divergence. Options include "JS" or "KL", with the default value being "KL."
<!-- For further detail, please refer to [Link](Divergence). -->

- ["JS"](Divergence)
- ["KL"](Divergence)


**discount**: A boolean indicating rank-awareness. By default, it is set to False, meaning no consideration of the position in the recommendation. If you wish to account for position, set it to True.

**n_bins**: (Optional) Integer. Determines the number of bins used for discretizing continuous data into intervals. The default value is 5.

**k**: (Optional) Integer. Rank cutoff @N. The default value is -1. When using default value -1, all items in the recommendation lists are evaluated.

## Formula

$Calibration=Cal(P^{\*}(c|H), Q^{\*}(c|R)) = \sum\limits_{c} Q^{\*}(c|R)f(\frac{P^{\*}(c|H)}{Q^{\*}(c|R)})$

The elements in the distribution Q are categories or complexities of recommendation items. The context P refers to the categories or complexities of the reading history.  
Here $P^{\*}(c|H)$ represents the rank-aware distribution of categories or complexity score bins (c) based on the users' reading history, with $Q^*(c|R)$, which represents the distribution in the recommendations provided to the user.

## Reference

Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September). RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).
