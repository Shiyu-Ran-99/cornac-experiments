## Goal of the metric

The Binomial metric can be employed in recommender system used for defining genre diversity that takes into account three key properties: genre coverage, genre redundancy and recommendation list size-awareness.
Genre coverage ensures that each genre is adequately represented in a recommendation list, taking into account user interests and genre specificity. This helps ensure that a diverse range of genres is included in the recommendations.

Redundancy is also taken into account, as it is crucial to avoid excessive representation of any particular genre, particularly in domains where items can belong to multiple genres. This helps prevent recommendations from being dominated by a single genre and promotes a more balanced and diverse set of suggestions.

Furthermore, recommendation list size-awareness recognizes the limitations of screen space when presenting recommendations. It considers how the size of the recommendation list impacts genre coverage and redundancy. This acknowledges that the number of items that can be displayed at once affects the diversity of genres that can be included and the potential for genre repetition within the list.

By considering these three factors, the Binomial metric provides a comprehensive evaluation of genre diversity in recommender systems.

## Description

To compute the Binomial metric, it is necessary to have information about the genres associated with each item. The item genres should be provided as a dictionary using the format {item idx: item genres}, where the item idx corresponds to the internal item id used by the Cornac system. The item genres should be represented as an ordered array, where a value of 0 indicates that the item does not belong to a certain category at the corresponding position in the genre array, and 1 indicates that the item belong to a certain category.

In most cases, genres within their respective domains are not completely separate or isolated categories For example, “The Lord of the Rings” by Tolkien can be classified as "Adventure", "Fiction", "High fantasy" and "British literature" all at once.

As an example, if the candidate genres array contains 8 genres, such as "Comedies," "Dramas," "Romance," "Action," "Adventure," "Fiction," "High fantasy," and "British literature," the representation of "The Lord of the Rings" would be [0, 0, 0, 0, 1, 1, 1, 1].

### Parameters

**item_genre**: A dictionary that maps item indices to the item genres array (numpy array).

**alpha**: (Optional) float between 0 and 1. A parameter to specify the weight of local probability when estimating probability $p_g$. The default value is 0.9.

**k**: (Optional) Integer. Rank cutoff @N. The default value is -1. When using default value -1, all items in the recommendation lists are evaluated.

## Formula

For a given item i and its associated set of genres G(i), the authors view the presence of a genre g within G(i) as a Bernoulli experiment. Calculating the number of items in a set S that belong to a specific genre g (i.e., the number of successes) can be denoted as:

$k_g^S =  |{i ∈ S : g ∈ G(i)}|$

Considering a recommendation list R with a size of N, the genre probability $P_{g}$ as an indicator of how "“adequate”" the number k<sub>g</sub><sup>R</sup> of items covering a genre g is in that recommendation.

**Estimate $P_{g}$**  Global genre distribution statistics and personalized user preferences are combined to estimate $P_g$. The generality of the genre can be estimated by the global proportion $P_{g}^{'}$ of items in the user preferences that cover it.

For a given user _u_, the personalized user preference is estimated by the local proportion $P_{g}^{''}$ of the items that user _u_ has interacted with. The formula to calculate $P_{g}$ is as follows:
 
$P_g = (1 - α) P_{g}^{'} + αP_{g}^{''}$

where $α$ is a parameter that specifies the weight of global probability and local probability.

**Coverage** The coverage score as the product of the probabilities of the genres not represented in the recommendation list, according to the distribution $X_g$. These probabilities are then normalized by the |G|-th root.

$Coverage(R) = Π_{g ∉ G(R)} P(X_g = 0)^{1/|G|}$

**Redundancy** The redundancy of a genre that appears k times in a recommendation list using a "remaining tolerance" score. This score indicates the probability of the genre appearing at least k times in a randomly generated list.
  
$P(X_g \ge k \mid X_g \gt 0) = 1 − \sum_{l=1}^{k-1} P(X_g = l \mid X_g \gt 0)$

The non-redundancy score is defined as the product of the "remaining tolerance" scores for each genre covered in the recommendation list. This score is then normalized by taking the |G(R)|-th root.

$NonRed(R) =\prod_{g \in G(R)} P(X_g \ge k_g^R \mid X_g \gt 0) ^{1/|G(R)|}$

**Size-awareness** The determination of coverage and redundancy in recommendations should depend on the length of the recommendation list. The Binomial metric considers the size of the recommendation list $N$. For instance, in the generation of a short recommendation list, the emphasis is placed on recommending items from the most relevant genres. Conversely, in a longer list, higher genre redundancy might be possible. 

**Binomial Metric** The Binomial metric incorporates user preferences through the parameter $p^{''}_{g}$. It effectively penalizes genres that are over-represented by significantly reducing their redundancy score. Additionally, it is designed to be adaptable to the length of the recommendation list, which is controlled by the parameter N.

$BinomDiv(R) = Coverage(R) · NonRed(R)$ 

## References

S. Vargas, L. Baltrunas, A. Karatzoglou, P. Castells
Coverage, redundancy and size-awareness in genre diversity for recommender systems
Proc. RecSys 2014, 8th ACM Conf. Recomm. Syst., Foster City, Silicon Valley, USA (2014), pp. 209-216
