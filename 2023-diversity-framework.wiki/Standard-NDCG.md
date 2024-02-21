## Standard nDCG
During the development of nDCG-related metrics, we discovered that Cornac already had an implemented nDCG metric. However, their implementation only considers binary relevance for positive and negative ground-truth items.  

To address this limitation, we introduced an alternative implementation in which ground-truth ratings are taken into account. To distinguish our version of nDCG, we named it "NDCG_score" in the diversity metric module, despite it not being a diversity metric per se.  
### Goal of the metric  
Normalized Discounted Cumulative Gain (nDCG) aims to assess the quality of rankings or the relevance of the top products listed. The underlying principle of NDCG is to prioritize more relevant products over irrelevant ones. A higher NDCG score signifies that the relevant products are ranked higher.  

### Description  
Cumulative Gain is the cumulative sum of the relevancy scores assigned to each product recommended by the system. It represents the total accumulated relevance or utility of the recommended products.  
Discounted Cumulative Gain (DCG) further considers rank by applying logarithmic discounting based on item position in the recommendation list. Items at the top receive higher weights, indicating their increased importance compared to those at the bottom. This weighting scheme captures the diminishing relevance of products as we move down the list.  
Relevancy indicates the degree of relevance an item holds for a user. For example, the ground truth ratings can be used.  

### Parameters  
**k**: (Optional) Integer. Rank cutoff @N. The default value is -1. When using default value -1, all items in the recommendation lists are evaluated.  

### Formula  
Considering each user u's "gain" $g_{u,i}$ from being recommended item i, the average Discounted Cumulative Gain (DCG) for a list of **J** items is defined as follows:  

$DCG = \frac{1}{N} \sum_{u=1}^{N} \sum_{j=1}^{J} \frac {g_{u, i_j}}{log_b(j+1)}$   

where $i_j$ is the item at position j in the list. The logarithm base is a free parameter, typically between 2 and 10. In this implementation, we use 2 as the logarithm base, to ensure all positions are discounted.  

**Ideal Discounted Cumulative Gain** $DCG^{\*}$: The ideal ordering is the ordering that maximizes cumulative gain at all levels.  

**NDCG**  $nDCG =\frac{DCG}{DCG^*}$  

## References  

C. Clarke, M. Kolla, G. Cormack, O. Vechtomova, A. Ashkan, S. Büttcher, I. MacKinnon
Novelty and diversity in information retrieval evaluation
Proc. 31st Int. ACM SIGIR Conf. Res. Dev. Infor. Retrieval (SIGIR ’08), Singapore, Singapore (2008), pp. 659-666.

Ricci, F., Rokach, L. and Shapira, B., 2015. Recommender systems: introduction and challenges. Recommender systems handbook, pp.1-34.
