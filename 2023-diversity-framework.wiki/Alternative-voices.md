## Goal of the metric

The Alternative voices metric aims to capture viewpoint diversity by measuring the proportional representation of individuals or organizations from minority (protected) and majority (unprotected) groups. These protected/unprotected groups can by identified through manual annotation or automatic extraction using categories such as gender and ethnicity. A alternative voice value of 0 signifies that the presence of minority and majority groups in the recommendations is proportionally equivalent to the overall pool. A higher score indicates greater divergence between the recommendation and the overall supply.

## Description

In order to compute the alternative voices metric, the required data consists of a dictionary with item indices as keys and a corresponding array of [minority score, majority score]. The item indices, represented as "item idx," refer to the internal item IDs utilized by the Cornac system. The minority and majority scores are calculated by evaluating the probability of the presence of minority and majority voices, respectively. Users are responsible for defining the minority and majority voices within news content. As an illustration, a 'minority voice' could be described as an individual detected through the NLP pipeline but without any association with a Wikipedia page. Please refer to the instance provided through our [data enhancement pipeline](Custom-Data-Enhancement.md). An array containing the minority score at position 0 and the majority score in the subsequent position should be prepared for each item.

### Parameters

**item_minor_major**: Dictionary. {item idx: [minority score, majority score]}. Item indices mapped to their respective minority score and majority score, which are saved in a numpy array.  

**data_type**: (Optional) string that specifies the data type. The choices are "gender", "ethnicity", and "mainstream". The default value is "mainstream".  

**divergence_type**: (Optional) string that determines the divergence method employed. The choices are "JS" or "KL". The default value is "KL."
For further detail, please refer to [Link](Divergence).

**discount**: (Optional) Boolean. By default, it is set to False, indicating no rank-awareness. If you want to consider position in the recommendation, set it to True.

**k**: (Optional) Integer. Rank cutoff @N. The default value is -1. When using default value -1, all items in the recommendation lists are evaluated.

## Formula

$AlternativeVoices=AltV(P(m|S),  Q^{\*}(m|R)) = \sum\limits_{m} Q^{\*}(m|R)f(\frac{P(m|S)}{Q^{\*}(m|R)})$

The elements in the distribution Q are minority and majority scores of recommendation items.
The context P refers to the minority and majority scores of the overall supply of available items in the data pool.  
M denotes the distribution of protected vs. non-protected groups, with m ∈ {Minority,Majority}. $P(m|S)$ and $Q^{\*}(m|S)$ refer to the distribution of these groups in the pool of available articles and rank-aware recommendation distribution respectively.

## Reference

Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September). RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).
