## Goal of the metric

Intra-list diversity (ILD) measures the diversity of a recommendation list.

## Description

The intra-list diversity of a set of recommended items is defined as the average **pairwise** distance of the items in the set.

The computation of ILD requires defining a distance measure d(i, j), a function of item features. Item_feature should be input using dictionary, i.e., {item idx: item feature vectors}.
This distance measure can be a configurable element.

The default distance type is cosine distance. Other distance measure can be correlation,Euclidean and Jaccard.

### Parameters

**item_feature**: A dictionary that maps item indices to their feature vectors. The feature vector should be provided using numpy array.  

**distance_type**: (Optional) String. ‘correlation’, ‘cosine’, ‘Euclidean’ or ‘Jaccard.’ By default, use cosine distance metric.

**k**: (Optional) Integer. Rank cutoff @N. The default value is -1. When using default value -1, all items in the recommendation lists are evaluated.

## Formula

$ILD(c_1, ..., c_n) = \frac {\sum_{i=1,...n}\sum_{j=1,...n}(1-similarity(c_i, c_j))}{\frac {n}{2} *(n-1)}$  

Here, the ILD diversity of a set of items, c1, ...cn, is defined as the average dissimilarity between all pairs of items in the recommendation list.

$1-similarity(c_i, c_j) =  diversity(c_i, c_j)$

## References

https://www.academia.edu/2655896/Improving_recommendation_diversity
