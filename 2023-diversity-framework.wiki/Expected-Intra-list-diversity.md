## Goal of the metric

Expected intra-list diversity (EILD) measures diversity within a recommendation list considering ranking position and relevance of the recommended items.

## Description

The expected intra-list diversity is defined as the average **pairwise** distance of the items in the recommendation list. A probabilistic recommendation browsing model is utilized to introduce item rank and relevance.
Three important concepts are introduced.
- _Discovery_ refers to whether an item is seen (or known) by a user.
- _Choice_ means whether an item is used, picked, selected, consumed, bought, etc., by a user.  
- _Relevance_ refers to whether an item is judged as relevant or not.

To calculate the Expected Intra-List Diversity (EILD), a distance measure, $dist(i, j)$, needs to be defined based on item features. The item features should be provided as a dictionary, denoted as {item idx: item feature vectors}. The default distance type currently used is cosine distance, where $dist(i, j)$ represents the cosine distance between items $i$ and $j$.

### Parameters

**item_feature**: A dictionary that maps item indices to their feature vectors. The feature vector should be provided using numpy array.

**disc_type**: (Optional) String. Type of the discount method used. Available choices are "exponential" or "logarithmic" or "reciprocal" or "nodiscount". Default value is "exponential."

**base**: (Optional) Float between 0 and 1. A probability to represent at each position in the ranked recommendation list, the user makes a decision whether or not to continue. Default base =0.9.

**k**: (Optional) Integer. Rank cutoff @N. The default value is -1. When using default value -1, all items in the recommendation lists are evaluated.

## Formula

$$ILD(i_k | u,R) = C_k^{'}\sum_{l}disc(l|k) p(rel|i_l,u)dist(i_k,i_l)$$  

Where $C_k^{'} = \frac {1} {\sum_{l \ne k}disc(l|k) p(rel|i_l,u)}$.  

$C_k^{'}$ normalizes the distance in the range [0, 1].

**Distance** $dist(i_k,i_l)$ can be measured by one minus the similarity between the items.  

I.e., $dist(i,j) = 1-sim(i,j)$.    

<!-- $sim(i,j) = q_i q_j / {\lVert q_i \rVert}{\lVert q_i \rVert}$. -->

**Discount**  
In case we consider an item in a recommendation list, the probability of been discovered, i.e. $p(seen|i,R)$, can be simplified by a decreasing discount function disc.
Item rank discounts are incorporated through a probabilistic recommendation browsing model. The underlying assumption is that users browse through the items in ranking order without skipping any, until they decide to stop browsing. At each position in the ranking, the user makes a binary decision of whether to continue or not, which can be modeled as a random variable.  
$disc(l|k) = disc(max(1, l-k))$  
Discount for an item at position $l$ knowing that position $k$ has been reached.
The implemented discount methods are:  

- Exponential ranking discount: $disc(k) = base^k$. (0<base<1)
- Reciprocal discount: $disc(k) = 1 / (k + 1)$.

* logarithmic discount: $disc(k) = 1 / log_2(k + 2)$.
* No discount: $disc(k) =1$.

**Relevance**   
If the available input consists of explicit user ratings, the probability of items being liked can be modeled by a heuristic mapping between rating values and probability of relevance.  

$$p(rel|i,u) =\frac { 2^{g(u,i)}-1} {2^{g_{max}}}$$ 
 
where $g$ is a utility function to be derived from ratings, e.g.,  

$$g(u,i) = max(0, r(u,i)-\tau )$$
 
where $\tau$ represents the “indifference” rating value.

$$EILD = C \sum_{ k} disc(k)p(rel|i_k,u)ILD(i_k|u,R)$$

## References

S. Vargas: New approaches to diversity and novelty in recommender systems. Proc. 4th BCS-IRSG Symp. Future Directions Inf. Access (FDIA 2011), Koblenz, Germany (2011), pp. 8-13.
