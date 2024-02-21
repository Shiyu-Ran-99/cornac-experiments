## Goal

Diversity is measured as the difference between two probability distributions: the difference in distribution between the issued recommendations (Q) and its reference probability distribution (P). Two types of divergence are implemented.

- KL divergence is a metric that quantifies the relative entropy between two probability distributions. It meets the condition of identity, meaning that if the KL divergence between two distributions, denoted as D(x,y), is zero, it implies that x and y are equal. It is non-symmetric. In other words, D(x, y) is not equal to D(y, x).
- The Jensen-Shannon (JS) divergence is based on the KL divergence. When using the base 2 logarithm, the JS divergence is bounded within the range of 0 to 1. Furthermore, JS divergence satisfies the properties of identity, symmetry, and the triangle inequality. The implementation refers to the square root of the JS formulation with a logarithm base 2.

## Formula

**KL divergence**  
The concept of relative entropy or KL (Kullback–Leibler) divergence between two probability mass functions P and Q (here, a recommendation and its context) is defined as:
$D_{KL}(P, Q) = - \sum\limits_{x \in \chi} P(x)log_2Q(x)+  \sum\limits_{x \in \chi} P(x)log_2P(x)$

**JS divergence**  
$D_{JS}(P, Q) = - \sum\limits_{x \in \chi} \frac {P(x)+ Q(x)}{2} log_2(\frac {P(x)+ Q(x)}{2}) + \frac {1}{2} \sum\limits_{x \in \chi} P(x)log_2P(x) +\frac {1}{2} \sum\limits_{x \in \chi} Q(x)log_2Q(x)$

**f divergence**  
A general formulation encompassing various divergence metrics, including JS and KL divergences is:
$D_{f}(P, Q) = \sum\limits_{x} Q(x)f(\frac {P(x)}{Q(x)})$,

where
$f_{JS}(t) = \frac {1}{2}  [(t+1)log(\frac{2}{t+1}) + tlog(t)]$, $f_{KL}(t) = tlog(t)$.

To avoid misspecified metrics, this metric computation uses the following formulas:  
$\overline{Q}(x) = (1 − α)Q(x) + αP(x)$,  
$\overline{P}(x) = (1 − α)P(x) + αQ(x)$.  
Here, α represents a small number that is close to zero. The purpose of α is to prevent artificially setting $D_{f}$ to zero in situations where a category, such as a news topic, is present in Q but not in P. Conversely, when a category is present in P but not in Q, the inclusion of α helps avoid division by zero in Q.

## Reference

Vrijenhoek, S., Bénédict, G., Gutierrez Granada, M., Odijk, D., & De Rijke, M. (2022, September). RADio–Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems (pp. 208-219).
