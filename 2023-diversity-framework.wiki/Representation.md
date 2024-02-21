## Goal of the metric

The Representation metric evaluates the viewpoint diversity (e.g. mentions of political topics or political parties), where the viewpoints are expressed categorically. It aims to quantify the presence of diverse perspectives within the recommended content. A score close to zero indicates a balanced representation, which means that the opinions shown in the recommendations are representative of those in society. A higher score indicates greater discrepancies in representation.

## Description

To calculate the Calibration metric, for each item, a list containing the presence of political actors should be provided. An _item_entities_ dictionary should be provided in the format {item idx: [item entities]}. For instance, if an article mentions political entities identified as "Democratic," "Republican," "Republican," and "Republican," the corresponding list would be ["Democratic", "Republican", "Republican", "Republican"]. The metric calculates the value by comparing the proportion of times that user u has encountered this perspective in their recommendations with the overall pool of articles.

### Parameters

**item_entities**: A dictionary that maps item indices to their respective list of entities. The entities should be Categorical.

**divergence_type**: (Optional) A string determining the employed method of divergence. Options include "JS" or "KL," with the default value being "KL."  
For further detail, please refer to [Link](Divergence).

**discount**: A boolean indicating rank-awareness. By default, it is set to False, meaning no consideration of the position in the recommendation. If you wish to account for position, set it to True.

**k**: (Optional) Integer. Rank cutoff @N. The default value is -1. When using default value -1, all items in the recommendation lists are evaluated.

## Formula

$Representation=Rep(P(p|S),  Q^{\*}(p|R)) = \sum\limits_{p} Q^{\*}(p|R)f(\frac{P(p|S)}{Q^{\*}(p|R)})$  
Here p represents the presence of a particular viewpoint, and $P(p|S)$ denotes the distribution of these viewpoints within the complete collection of articles.
$Q^{\*}(p|R))$ represents the rank-sensitive distribution of viewpoints within the set of recommended articles.

## Reference

Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September). RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).
