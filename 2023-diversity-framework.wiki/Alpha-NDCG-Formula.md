### Formula Part 2  
**Estimate the probability**   
Conventionally, probabilities are often assigned as either 0 or 1 to specific instances of *u* and *d*. For example, $P(n_i \in u) = 1$ implies that nugget $n_i$ is confirmed to be a part of *u*, while $P(n_i \in u) = 0$ signifies that $n_i$ is confirmed not to be a part of *u*. Similarly, $P(n_i \in d) = 1$ implies $n_i$ is present in *d*, and vice versa. However, this traditional approach tends to overestimate the certainty with which these values can be determined. To better reflect the actual scenario, the paper by Clarke et al. (2008) adopted a more flexible perspective. The paper made an assumption where a human assessor reads a document *d* and makes a binary decision about each nugget $n_i$ contained in the document. If the assessor judges that *d* contains $n_i$, it is denoted as $J(d, i) = 1$; if not, it is denoted as $J(d, i) = 0$.  
$$
P(n_i \in d)=
\begin{cases}
\alpha& \text{if } J(d, i) = 1,\\
0& \text{otherwise.}
\end{cases}
$$ 
$\alpha$ holds a value where $0 \lt \alpha \leqslant 1$, which implies the potential for assessor error. This definition means that positive judgments can be mistaken, while negative judgments are consistently accurate. 

The next step is to estimate $P(n_i \in u)$. The user preferences are often derived from explicit or implicit user interactions and feedback. The paper makes an assumption that nuggets are independent and equally likely to be relevant, which results in $P(n_i \in u) = \gamma$ for all *i*. 
With these two assumptions, the previous equation becomes:  
$$P(R=1 | u, d) = 1 - \prod_{i=1}^m(1 - \gamma \alpha J(d, i))$$ 

**Redundancy and Novelty**   
With the equation of $P(R=1\mid u, d)$ presented earlier, a single document can be analyzed. However, when dealing with the second and subsequent documents, their relevance needs to be considered relative to those ranked higher.  
Now the goal is to evaluate the relevance of $d_k$ in a ranked list, suppose the relevance estimates for the initial *k-1* documents $(d_1,...,d_{k-1})$ are known.  
Let $R_1,...,R_k$ represent the random variables associated with relevance at each rank. 
Our objective is to estimate the probability $P(R_k=1 | u, d_1, ..., d_{k})$.  
To emphasize novelty over redundancy, an assumption is made that if a particular nugget is already present in the initial *k-1* documents, its repetition in $d_k$ won't offer any extra value.  Consequently, the probability of the user still maintaining interest in a nugget $n_i$ depends on the contents of the initial $k-1$ documents.  
$$P(n_i \in u \mid d_1,...,d_{k-1}) = P(n_i \in u) \prod_{j=1}^{k-1} P(n_i \notin d_j)$$
Define $r_i$ as the number of documents evaluated up to position *k-1* which have been determined to contain the nugget $n_i$. 
$r_{i, k-1}$ is calculated as follows:   
$$r_{i, k-1} = \sum_{j=1}^{k-1}J(d_j, i)$$,
$$r_{i, 0} = 0$$  
Thus 
$$\prod_{j=1}^{k-1}P(n_i \notin d_j) = (1 - \alpha)^{r_{i,k-1}}$$  
And in place of the equation we have earlier,  
$$P(R_k = 1|u,d_1,...,d_k) = 1- \prod_{i=1}^{m}(1-\gamma \alpha J(d_k,i)(1-\alpha)^{r_{i,k-1}})$$

**Cumulative Gain Measures**  
The paper by Clarke et al. (2008) further simplifies the equation as follows:  
$$P(R_k = 1|u,d_1,...,d_k) \approx \gamma \alpha \sum_{i=1}^{m} J(d_k,i)(1-\alpha)^{r_{i,k-1}}$$
By dropping the constant $\gamma \alpha$, the $k$-th element of the gain vector G can be defined as:  
$$G[k] = \sum_{j=1}^{m} J(d_k, i) (1 - \alpha)^{r_{i,k-1}}$$   
The cumulative gain vector CG can be defined as: 
$$CG[k] = \sum_{j=1}^{k} G[j]$$  
By taking a typical logarithm discount, the discounted cumulative gain is: 
$$DCG[k] = \sum_{j=1}^{k} G[j]/(log_2(1+j))$$   
The ideal gain $DCG^{\*}$ can be obtained by ranking the results in descending order of their relevance.  
$\alpha$-nDCG can be computed as:  
$$\alpha-nDCG[k] = \frac{DCG[k]}{DCG^{*}[k]}$$

When $\alpha = 0$, the $\alpha-nDCG$ measure aligns with the standard nDCG, where the number of matching nuggets is used as the ground-truth relevance value.   


## References  

C. Clarke, M. Kolla, G. Cormack, O. Vechtomova, A. Ashkan, S. Büttcher, I. MacKinnon
Novelty and diversity in information retrieval evaluation
Proc. 31st Int. ACM SIGIR Conf. Res. Dev. Infor. Retrieval (SIGIR ’08), Singapore, Singapore (2008), pp. 659-666.

Ricci, F., Rokach, L. and Shapira, B., 2015. Recommender systems: introduction and challenges. Recommender systems handbook, pp.1-34.