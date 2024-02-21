## alpha-NDCG
### Goal of the metric
The $\alpha$-nDCG metric proposed by Clarke et al. (2008) integrated diversity and novelty into a novel measure of retrieved document relevancy, leveraging the foundation of the Normalized Discounted Cumulative Gain(nDCG). In this project, we implemented and integrated the $\alpha$-nDCG metric into Cornac. To have an overview of the standard nDCG metric, which serves as the basis of this $\alpha$-nDCG metric, please refer to [nDCG](Standard-NDCG).  
Similar to the information retrieval task, recommender systems aim to produce a ranked list that considers the breadth of available information. Ideally, the ordering of recommended items should reflect the preferences of users. Top-ranked items should cover fundamental aspects of each topic that align with users' interests. Subsequent documents would then complement this core information, avoiding repetitive content. This metric offers a method for rewarding novelty, aimed at reducing repetition.  Additionally, it considers diversity, which aims to tackle uncertainty and offer diverse interpretations. As the parameter $\alpha$ increases, the metric places more emphasis on rewarding novelty. A higher $\alpha$-nDCG value indicates better ranking quality and relevance of recommendations, taking into account both diversity and novelty within the list.  

### Description   
In our implementation, to determine relevance, we utilized genres from the user's historical data. If a recommended item shares a genre with the user's history, it is considered relevant. We made an assumption that genres are independent and those genres appear in the user history are equally likely to be relevant.    
To compute the $\alpha$-nDCG metric, the item genres should be provided as a dictionary using the format {item idx: item genres}, where the item idx corresponds to the internal item id used by the Cornac system. The item genres should be represented as an ordered array, where a value of 0 indicates that the item does not belong to a certain category at the corresponding position in the genre array, and 1 indicates that the item belong to a certain category.   
For example, “The Lord of the Rings” by Tolkien can be classified as "Adventure", "Fiction", "High fantasy" and "British literature" all at once. If the candidate genres array contains 8 genres, such as "Comedies," "Dramas," "Romance," "Action," "Adventure," "Fiction," "High fantasy," and "British literature," the representation of "The Lord of the Rings" would be [0, 0, 0, 0, 1, 1, 1, 1].  

### Parameters   
**item_genre**: A dictionary that maps item indices to the item genres array(numpy array).

**k**: (Optional) Integer. Rank cutoff @N. The default value is -1. When using default value -1, all items in the recommendation lists are evaluated. 

**alpha**: (Optional) float between 0 and 1. A parameter to reward novelty. When the parameter "alpha" is set to 0, the $\alpha$-nDCG metric aligns with the standard nDCG. The default value is 0.5.

### Formula Part 1
In the paper by Clarke et al. (2008), a document is considered relevant if it includes at least one nugget that matches a nugget in the user's information requirement.  
Consider $R$ as a binary random variable denoting relevance. If the user's information requirement as a set of nuggets, denoted as $u \subseteq N$, where $N = \{n_1,..., n_m\}$ represents the range of potential nuggets. The information contained within a document can also be represented as a set of nuggets, denoted as $d \subseteq N$.  
The document is considered as relevant if it includes at least one nugget that aligns with the user's information requirement, which also means:  
$P(R=1 | u, d) = P(\exists n_i \text{ such that } n_i \in u \cap d)$          
Under the assumption that $n_i \subseteq u$, $n_{j \neq i} \subseteq u$,  $n_i \subseteq d$, and $n_{j \neq i} \subseteq d$ are independent, the equation above can be written as:    
$P(R=1 | u, d) = 1 - \prod_{i=1}^m(1 - P(n_i \in u) \cdot P(n_i \in d))$  
Here, strong assumptions are made that a user's interest in one piece of information is not influenced by their interest in other pieces of information.  

Due to the maximum limit on formula rendering in GitLab, a separate page has been created for the remaining formula description. For detailed information, kindly visit the [alpha-nDCG Formula Part2](Alpha-NDCG-Formula) page.
<!-- ### Formula   
In the paper by Clarke et al. (2008), a document is considered relevant if it includes at least one nugget that matches a nugget in the user's information requirement.  
Consider $R$ as a binary random variable denoting relevance. If the user's information requirement as a set of nuggets, denoted as $u \subseteq N$, where $N = \{n_1,..., n_m\}$ represents the range of potential nuggets. The information contained within a document can also be represented as a set of nuggets, denoted as $d \subseteq N$.  
The document is considered as relevant if it includes at least one nugget that aligns with the user's information requirement, which also means:  
$P(R=1 | u, d) = P(\exists n_i \text{ such that } n_i \in u \cap d)$          
Under the assumption that $n_i \subseteq u$, $n_{j \neq i} \subseteq u$,  $n_i \subseteq d$, and $n_{j \neq i} \subseteq d$ are independent, the equation above can be written as:    
$P(R=1 | u, d) = 1 - \prod_{i=1}^m(1 - P(n_i \in u) \cdot P(n_i \in d))$  
Here, strong assumptions are made that a user's interest in one piece of information is not influenced by their interest in other pieces of information.   
**Estimate the probability $P(n_i \in d)$ and $P(n_i \in u)$**   
Conventionally, probabilities are often assigned as either 0 or 1 to specific instances of $u$ and $d$. For example, $P(n_i \in u) = 1$ implies that $n_i$ is confirmed to be a part of $u$, while $P(n_i \in u) = 0$ signifies that $n_i$ is confirmed not to be a part of $u$. Similarly, $P(n_i \in d) = 1$ implies $n_i$ is present in $d$, and vice versa. However, this traditional approach tends to overestimate the certainty with which these values can be determined. To better reflect the actual scenario, the paper adopted a more flexible perspective. The paper made an assumption where a human assessor reads a document $d$ and makes a binary decision about each nugget $n_i$ contained in the document. If the assessor judges that $d$ contains $n_i$, it is denoted as $J(d, i) = 1$; if not, it is denoted as $J(d, i) = 0$.  
$$
P(n_i \in d)=
\begin{cases}
\alpha& \text{if } J(d, i) = 1,\\
0& \text{otherwise.}
\end{cases}
$$ 
The $\alpha$ holds a value where $0 \lt \alpha \leqslant 1$, which implies the potential for assessor error. This definition means that positive judgments can be mistaken, while negative judgments are consistently accurate. 

The next step is to estimate $P(n_i \in u)$. The user preferences are often derived from explicit or implicit user actions and feedback. The paper makes an assumption that nuggets are independent and equally likely to be relevant, which results in $P(n_i \in u) = \gamma$. 
With these two assumptions, the previous equation becomes:  
$P(R=1 | u, d) = 1 - \prod_{i=1}^m(1 - \gamma \alpha J(d, i))$  

**Redundancy and Novelty**   
With the equation of $P(R=1\mid u, d)$ presented earlier, a single document can be analyzed. However, when dealing with the second and subsequent documents, their relevance needs to be considered relative to those ranked higher.  
Now the goal is to evaluate the relevance of $d_k$ in a ranked list, suppose the relevance estimates for the initial $k-1$ documents $(d_1,...,d_{k-1})$ are known.  
Let $R_1,...,R_k$ represent the random variables associated with relevance at each rank. 
Our objective is to estimate the probability $P(R_k=1 | u, d_1, ..., d_{k})$.  
To emphasize novelty over redundancy, an assumption is made that if a particular nugget is already present in the initial $k-1$ documents, its repetition in $d_k$ won't offer any extra value.  Consequently, the probability of the user still maintaining interest in a nugget $n_i$ depends on the contents of the initial $k-1$ documents.  
$$P(n_i \in u \mid d_1,...,d_{k-1}) = P(n_i \in u) \prod_{j=1}^{k-1} P(n_i \notin d_j)$$
Define $r_i$ as the number of documents evaluated up to position $k-1$ which have been determined to contain the nugget $n_i$. 
$r_{i, k-1}$ is calculated as follows:   
$r_{i, k-1} = \sum_{j=1}^{k-1}J(d_j, i)$
$r_{i, 0} = 0$  
Thus 
$$\prod_{j=1}^{k-1}P(n_i \notin d_j) = (1 - \alpha)^{r_{i,k-1}}$$  
And in place of the equation we have earlier,  
$$P(R_k = 1|u,d_1,...,d_k) = 1- \prod_{i=1}^{m}(1-\gamma \alpha J(d_k,i)(1-\alpha)^{r_{i,k-1}})$$

**Cumulative Gain Measures**  
By further simplifying the equation as follows:  
$$P(R_k = 1|u,d_1,...,d_k) \approx \gamma \alpha \sum_{i=1}^{m} J(d_k,i)(1-\alpha)^{r_{i,k-1}}$$
By dropping the constant $\gamma \alpha$, the $k$-th element of the gain vector G can be defined as:  
$$G[k] = \sum_{j=1}^{m} J(d_k, i) (1 - \alpha)^{r_{i,k-1}}$$   
The cumulative gain vector CG can be defined as: 
$$CG[k] = \sum_{j=1}^{k} G[j]$$  
By taking a typical discount $log_2(1+k)$, the discounted cumulative gain is: 
$$DCG[k] = \sum_{j=1}^{k} G[j]/(log_2(1+j))$$   
The ideal gain $DCG^{*}$ can be obtained by ranking the results in descending order of their relevance.  
$\alpha$-nDCG can be computed as:  
$$\alpha-nDCG[k] = \frac{DCG[k]}{DCG^{*}[k]}$$

When $\alpha = 0$, the $\alpha-nDCG$ measure aligns with the standard nDCG, where the number of matching nuggets is used as the ground-truth relevance value.    -->

## References  

C. Clarke, M. Kolla, G. Cormack, O. Vechtomova, A. Ashkan, S. Büttcher, I. MacKinnon
Novelty and diversity in information retrieval evaluation
Proc. 31st Int. ACM SIGIR Conf. Res. Dev. Infor. Retrieval (SIGIR ’08), Singapore, Singapore (2008), pp. 659-666.

Ricci, F., Rokach, L. and Shapira, B., 2015. Recommender systems: introduction and challenges. Recommender systems handbook, pp.1-34.
