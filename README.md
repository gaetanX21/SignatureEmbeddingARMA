# SignatureEmbeddingARMA
Illustrating the relevance of the Signature as an abstract embedding for path data.

## Signature embeddings for machine learning

### A. Mathematical definition
Given a smooth path $X\in \mathcal{C}([0,T],\R^d)$, one can compute its signature, which is an object living $T(\R^d)$ (the tensor algebra of $\R^d$) and which encodes geometric properties of $X$. More precisely, the signature of $X$ is defined as
$$\mathbb{X}=(1,\mathbb{X}^1,\mathbb{X}^2,\ldots,\mathbb{X}^k,\ldots)$$
where $$\mathbb{X}^k=\int_{0<u_1<\ldots<u_k<T} dX_{u_1}\otimes \cdots \otimes dX_{u_k} \in \big(\R^d\big)^{\otimes k}$$ is a tensor of rank $k$ over the vector space $\R^d$ i.e. $$\mathbb{X}^k=(\mathbb{X}^{i_1,\ldots,i_k})_{1\leq i_1,\ldots,i_k\leq d} \text{ where } \mathbb{X}^{i_1,\ldots,i_k} = \int_{0<u_1<\ldots<u_k<T} dX_{u_1}^{i_1}\ldots dX_{u_k}^{i_k} \in \R$$

Note that the signature is an infinite series of increasingly large tensors. In practice, when computing the signature we restrict ourselves to the tensors of rank less or equal to $N$, which is the depth chosen. We call the resulting object the $N$-truncated signature of $X$ and denote it $\mathbb{X}^{\leq N}$. Thus $\mathbb{X}^{\leq N}=(1,\mathbb{X}^1,\mathbb{X}^1,\ldots,\mathbb{X}^N)$.

### B. Use in machine learning
It is common to work with path data in machine learning, especially in the form of time series. Also, non-path data can easily be turned into path data. Therefore, designing good features when dealing with path data is paramount. In particular, one can use low-rank signature terms as features to characterize path data.

Note that several transforms can be applied to the path before computing its signature and using the terms as features. In particular, the Lead-Lag transform, which turns a signal $X_t\in\R^d$ into a signal $X_t^{LL}=(X_t^\text{lead},X_t^\text{lag}) \in\R^{2d}$ is quite classical in machine learning and usually dramatically improves performance.

### C. Use case
In this notebook we will try to demonstrate the relevance of signature terms as features for path data. To do so, we will generate samples of two very similar (at least from the human eye) ARMA processes and try to classify them using two approaches:

1) a neural network taking as input all the points $\{X_{t_i}\}_{1\leq i \leq n}$ where $\{t_i\}$_{1\leq i \leq n}$ is the discretization scheme
2) a logistic regression model taking as input low-rank signature terms

In particular, we will compare the performance of these two classifiers for various values of $n$.
