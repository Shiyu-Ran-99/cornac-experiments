## Goal of the metric

The Fragmentation metric assesses the differences in recommended news story chains, which are sets of articles that cover the same issue or event from various perspectives, writing styles, or temporal contexts. By comparing these differences among users, Fragmentation measures the level of overlap between news story chains among users, revealing the existence of a shared public sphere or individual information bubbles.

In theory, assessing Fragmentation metric would involve calculating the divergence between the recommendations received by a user and all others, but this is impractical with large datasets.Instead, users can specify the number of users to compare (minimum 1), and a random sampling strategy is employed to select user pairs for Fragmentation evaluation.
A Fragmentation score of 0 signifies a complete overlap among users, while a score of 1 indicates no overlap.

## Description

This metric requires that individual articles be aggregated into higher-level news story chains over time. The fragmentation value is then determined by calculating the average divergence between the distribution of recommendation stories for a given user and the distribution of recommendation stories across all sampled users.

## Parameters

**item_story**: A dictionary that maps item indices saved in cornac to their respective story chains. The stories are categorical values.

**divergence_type**: (Optional) A string determining the employed method of divergence. Options include "JS" or "KL", with the default value being "KL."
For further detail, please refer to [Link](Divergence).  
**discount**: A boolean indicating rank-awareness. By default, it is set to False, meaning no consideration of the position in the recommendation. If you wish to account for position, set it to True.

**n_samples**: (Optional) Integer. The number of users to compare (minimum 1). By default n_samples=1.

**k**: (Optional) Integer. Rank cutoff @N. The default value is -1. When using default value -1, all items in the recommendation lists are evaluated.

## Formula

$Fragmentation=Frag(P^{\*}(e|R^{u}),  Q^{\*}(e|R^{v})) = \sum\limits_{e} Q^{\*}(e|R^{v})f(\frac{P^{\*}(e|R^{u})}{Q^{\*}(e|R^{v})})$

The elements within the distribution Q correspond to stories of recommendation items, while the context P pertains to the stories of recommendation items for other users.  
$P^{\*}(e|R^{u})$ represents the rank-aware distribution of news events e across the recommendations R for user u, while $Q^{\*}(e|R^{v})$ represents the corresponding distribution for user v.

## Reference

Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September). RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).
