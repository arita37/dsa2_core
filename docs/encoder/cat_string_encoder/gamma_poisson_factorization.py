import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _k_init
from sklearn.decomposition.nmf import _beta_divergence
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils.extmath import row_norms, safe_sparse_dot


class OnlineGammaPoissonFactorization(BaseEstimator, TransformerMixin):
    """
    Online Non-negative Matrix Factorization by minimizing the
    Kullback-Leibler divergence.

    Parameters
    ----------

    n_topics: int, default=10
        Number of topics of the matrix Factorization.

    batch_size: int, default=100

    gamma_shape_prior: float, default=1.1
        Shape parameter for the Gamma prior distribution.

    gamma_scale_prior: float, default=1.0
        Shape parameter for the Gamma prior distribution.

    r: float, default=1
        Weight parameter for the update of the W matrix

    hashing: boolean, default=False
        If true, HashingVectorizer is used instead of CountVectorizer.

    hashing_n_features: int, default=2**10
        Number of features for the HashingVectorizer. Only relevant if
        hashing=True.

    hashing: boolean, default=True
        If true, the weight matrix W is rescaled at each iteration
        to have an l1 norm equal to 1 for each row.

    analizer: str, default 'char'.
        Analizer parameter for the CountVectorizer/HashingVectorizer.
        Options: {‘word’, ‘char’, ‘char_wb’}

    tol: float, default=1e-3
        Tolerance for the convergence of the matrix W

    mix_iter: int, default=2

    max_iter: int, default=10

    ngram_range: tuple, default=(2, 4)

    init: str, default 'k-means++'
        Initialization method of the W matrix.

    random_state: default=None

    Attributes
    ----------

    References
    ----------
    """

    def __init__(
        self,
        n_topics=10,
        batch_size=512,
        gamma_shape_prior=1.1,
        gamma_scale_prior=1.0,
        r=0.7,
        rho=0.99,
        hashing=False,
        hashing_n_features=2 ** 12,
        init="k-means++",
        tol=1e-4,
        min_iter=2,
        max_iter=5,
        ngram_range=(2, 4),
        analizer="char",
        add_words=False,
        random_state=None,
        rescale_W=True,
        max_iter_e_step=20,
    ):

        self.ngram_range = ngram_range
        self.n_topics = n_topics
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.r = r
        self.rho = rho
        self.batch_size = batch_size
        self.tol = tol
        self.hashing = hashing
        self.hashing_n_features = hashing_n_features
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.init = init
        self.analizer = analizer
        self.add_words = add_words
        self.random_state = check_random_state(random_state)
        self.rescale_W = rescale_W
        self.max_iter_e_step = max_iter_e_step

        if self.hashing:
            self.ngrams_count = HashingVectorizer(
                analyzer=self.analizer,
                ngram_range=self.ngram_range,
                n_features=self.hashing_n_features,
                norm=None,
                alternate_sign=False,
            )
            if self.add_words:
                self.word_count = HashingVectorizer(
                    analyzer="word",
                    n_features=self.hashing_n_features,
                    norm=None,
                    alternate_sign=False,
                )
        else:
            self.ngrams_count = CountVectorizer(
                analyzer=self.analizer, ngram_range=self.ngram_range
            )
            if self.add_words:
                self.word_count = CountVectorizer()

    def _update_H_dict(self, X, H):
        for x, h in zip(X, H):
            self.H_dict[x] = h

    def _init_vars(self, X):
        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_V = self.ngrams_count.fit_transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count.fit_transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format="csr")

        if not self.hashing:
            self.vocabulary = self.ngrams_count.get_feature_names()
            if self.add_words:
                self.vocabulary = np.concatenate(
                    (self.vocabulary, self.word_count.get_feature_names())
                )

        _, self.n_vocab = unq_V.shape
        self.W_, self.A_, self.B_ = self._init_w(unq_V[lookup], X)
        unq_H = _rescale_h(unq_V, np.ones((len(unq_X), self.n_topics)))
        self.H_dict = dict()
        self._update_H_dict(unq_X, unq_H)
        if self.rho is None:
            self.rho_ = self.r ** (self.batch_size / len(X))
        elif self.r is None:
            self.rho_ = self.rho
        else:
            raise AttributeError
        return unq_X, unq_V, lookup

    def _get_H(self, X):
        H_out = np.empty((len(X), self.n_topics))
        for x, h_out in zip(X, H_out):
            h_out[:] = self.H_dict[x]
        return H_out

    def _init_w(self, V, X):
        if self.init == "k-means++":
            W = (
                _k_init(
                    V,
                    self.n_topics,
                    row_norms(V, squared=True),
                    random_state=self.random_state,
                    n_local_trials=None,
                )
                + 0.1
            )
        elif self.init == "random":
            W = self.random_state.gamma(
                shape=self.gamma_shape_prior,
                scale=self.gamma_scale_prior,
                size=(self.n_topics, self.n_vocab),
            )
        elif self.init == "k-means":
            prototypes = get_kmeans_prototypes(X, self.n_topics, random_state=self.random_state)
            W = self.ngrams_count.transform(prototypes).A + 0.1
            if self.add_words:
                W2 = self.word_count.transform(prototypes).A + 0.1
                W = np.hstack((W, W2))
            # if k-means doesn't find the exact number of prototypes
            if W.shape[0] < self.n_topics:
                W2 = (
                    _k_init(
                        V,
                        self.n_topics - W.shape[0],
                        row_norms(V, squared=True),
                        random_state=self.random_state,
                        n_local_trials=None,
                    )
                    + 0.1
                )
                W = np.concatenate((W, W2), axis=0)
        else:
            raise AttributeError("Initialization method %s does not exist." % self.init)
        W /= W.sum(axis=1, keepdims=True)
        A = np.ones((self.n_topics, self.n_vocab)) * 1e-10
        B = A.copy()
        return W, A, B

    def fit(self, X, y=None):
        """Fit the OnlineGammaPoissonFactorization to X.

        Parameters
        ----------
        X : string aself.rrray-like, shape [n_samples, n_features]
            The data to determine the categories of each feature
        Returns
        -------
        self
        """
        assert X.ndim == 1
        unq_X, unq_V, lookup = self._init_vars(X)
        n_batch = (len(X) - 1) // self.batch_size + 1
        del X
        unq_H = self._get_H(unq_X)

        for iter in range(self.max_iter):
            for i, (unq_idx, idx) in enumerate(batch_lookup(lookup, n=self.batch_size)):
                if i == n_batch - 1:
                    W_last = self.W_.copy()
                unq_H[unq_idx] = _multiplicative_update_h(
                    unq_V[unq_idx],
                    self.W_,
                    unq_H[unq_idx],
                    epsilon=1e-3,
                    max_iter=self.max_iter_e_step,
                    rescale_W=self.rescale_W,
                    gamma_shape_prior=self.gamma_shape_prior,
                    gamma_scale_prior=self.gamma_scale_prior,
                )
                _multiplicative_update_w(
                    unq_V[idx], self.W_, self.A_, self.B_, unq_H[idx], self.rescale_W, self.rho_
                )

                if i == n_batch - 1:
                    W_change = np.linalg.norm(self.W_ - W_last) / np.linalg.norm(W_last)

            if (W_change < self.tol) and (iter >= self.min_iter - 1):
                break

        self._update_H_dict(unq_X, unq_H)
        return self

    def get_feature_names(self, n_top=3):
        vectorizer = CountVectorizer()
        vectorizer.fit(list(self.H_dict.keys()))
        vocabulary = np.array(vectorizer.get_feature_names())
        encoding = self.transform(np.array(vocabulary).reshape(-1))
        encoding = abs(encoding)
        encoding = encoding / np.sum(encoding, axis=1, keepdims=True)
        n_components = encoding.shape[1]
        topic_labels = []
        for i in range(n_components):
            x = encoding[:, i]
            labels = vocabulary[np.argsort(-x)[:n_top]]
            topic_labels.append(labels)
        topic_labels = [", ".join(label) for label in topic_labels]
        return topic_labels

    def score(self, X):
        """
        Returns the Kullback-Leibler divergence.

        Parameters
        ----------
        X : array-like (str), shape [n_samples,]
            The data to encode.

        Returns
        -------
        kl_divergence : float.
            Transformed input.
        """

        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_V = self.ngrams_count.transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format="csr")

        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        for slice in gen_batches(n=unq_H.shape[0], batch_size=self.batch_size):
            unq_H[slice] = _multiplicative_update_h(
                unq_V[slice],
                self.W_,
                unq_H[slice],
                epsilon=1e-3,
                max_iter=self.max_iter_e_step,
                rescale_W=self.rescale_W,
                gamma_shape_prior=self.gamma_shape_prior,
                gamma_scale_prior=self.gamma_scale_prior,
            )
        kl_divergence = _beta_divergence(
            unq_V[lookup], unq_H[lookup], self.W_, "kullback-leibler", square_root=False
        )
        return kl_divergence

    def partial_fit(self, X, y=None):
        assert X.ndim == 1
        if hasattr(self, "vocabulary"):
            unq_X, lookup = np.unique(X, return_inverse=True)
            unq_V = self.ngrams_count.transform(unq_X)
            if self.add_words:
                unq_V2 = self.word_count.transform(unq_X)
                unq_V = sparse.hstack((unq_V, unq_V2), format="csr")

            unseen_X = np.setdiff1d(unq_X, np.array([*self.H_dict]))
            unseen_V = self.ngrams_count.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format="csr")

            if unseen_V.shape[0] != 0:
                unseen_H = _rescale_h(unseen_V, np.ones((len(unseen_X), self.n_topics)))
                for x, h in zip(unseen_X, unseen_H):
                    self.H_dict[x] = h
                del unseen_H
            del unseen_X, unseen_V
        else:
            unq_X, unq_V, lookup = self._init_vars(X)
            self.rho_ = self.rho

        unq_H = self._get_H(unq_X)
        unq_H = _multiplicative_update_h(
            unq_V,
            self.W_,
            unq_H,
            epsilon=1e-3,
            max_iter=self.max_iter_e_step,
            rescale_W=self.rescale_W,
            gamma_shape_prior=self.gamma_shape_prior,
            gamma_scale_prior=self.gamma_scale_prior,
        )
        self._update_H_dict(unq_X, unq_H)
        _multiplicative_update_w(
            unq_V[lookup], self.W_, self.A_, self.B_, unq_H[lookup], self.rescale_W, self.rho_
        )
        return self

    def _add_unseen_keys_to_H_dict(self, X):
        unseen_X = np.setdiff1d(X, np.array([*self.H_dict]))
        if unseen_X.size > 0:
            unseen_V = self.ngrams_count.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format="csr")

            unseen_H = _rescale_h(unseen_V, np.ones((unseen_V.shape[0], self.n_topics)))
            self._update_H_dict(unseen_X, unseen_H)

    def transform(self, X):
        """Transform X using the trained matrix W.

        Parameters
        ----------
        X : array-like (str), shape [n_samples,]
            The data to encode.

        Returns
        -------
        X_new : 2-d array, shape [n_samples, n_topics]
            Transformed input.
        """
        unq_X = np.unique(X)
        unq_V = self.ngrams_count.transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format="csr")

        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        for slice in gen_batches(n=unq_H.shape[0], batch_size=self.batch_size):
            unq_H[slice] = _multiplicative_update_h(
                unq_V[slice],
                self.W_,
                unq_H[slice],
                epsilon=1e-3,
                max_iter=100,
                rescale_W=self.rescale_W,
                gamma_shape_prior=self.gamma_shape_prior,
                gamma_scale_prior=self.gamma_scale_prior,
            )
        self._update_H_dict(unq_X, unq_H)
        return self._get_H(X)


def _rescale_W(W, A, B):
    s = W.sum(axis=1, keepdims=True)
    W /= s
    A /= s
    return W, A, B


def _multiplicative_update_w(Vt, W, A, B, Ht, rescale_W, rho):
    A *= rho
    A += W * safe_sparse_dot(Ht.T, Vt.multiply(np.dot(Ht, W) ** -1))
    B *= rho
    B += Ht.sum(axis=0).reshape(-1, 1)
    np.divide(A, B, out=W)
    if rescale_W:
        _rescale_W(W, A, B)
    return W, A, B


def _rescale_h(V, H):
    epsilon = 1e-10  # in case of a document having length=0
    H *= np.maximum(epsilon, V.sum(axis=1).A)
    H /= H.sum(axis=1, keepdims=True)
    return H


def _multiplicative_update_h(
    Vt,
    W,
    Ht,
    epsilon=1e-3,
    max_iter=10,
    rescale_W=False,
    gamma_shape_prior=1.1,
    gamma_scale_prior=1.0,
):
    if rescale_W:
        WT1 = 1 + 1 / gamma_scale_prior
        W_WT1 = W / WT1
    else:
        WT1 = np.sum(W, axis=1) + 1 / gamma_scale_prior
        W_WT1 = W / WT1.reshape(-1, 1)
    const = (gamma_shape_prior - 1) / WT1
    squared_epsilon = epsilon ** 2
    for vt, ht in zip(Vt, Ht):
        vt_ = vt.data
        idx = vt.indices
        W_WT1_ = W_WT1[:, idx]
        W_ = W[:, idx]
        squared_norm = 1
        for n_iter_ in range(max_iter):
            if squared_norm <= squared_epsilon:
                break
            aux = np.dot(W_WT1_, vt_ / np.dot(ht, W_))
            ht_out = ht * aux + const
            squared_norm = np.dot(ht_out - ht, ht_out - ht) / np.dot(ht, ht)
            ht[:] = ht_out
    return Ht


def batch_lookup(lookup, n=1):
    len_iter = len(lookup)
    for idx in range(0, len_iter, n):
        indices = lookup[slice(idx, min(idx + n, len_iter))]
        unq_indices = np.unique(indices)
        yield (unq_indices, indices)


def get_kmeans_prototypes(
    X,
    n_prototypes,
    hashing_dim=128,
    ngram_range=(2, 4),
    sparse=False,
    sample_weight=None,
    random_state=None,
):
    """
    Computes prototypes based on:
      - dimensionality reduction (via hashing n-grams)
      - k-means clustering
      - nearest neighbor
    """
    vectorizer = HashingVectorizer(
        analyzer="char",
        norm=None,
        alternate_sign=False,
        ngram_range=ngram_range,
        n_features=hashing_dim,
    )
    projected = vectorizer.transform(X)
    if not sparse:
        projected = projected.toarray()
    kmeans = KMeans(n_clusters=n_prototypes, random_state=random_state)
    kmeans.fit(projected, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    neighbors = NearestNeighbors()
    neighbors.fit(projected)
    indexes_prototypes = np.unique(neighbors.kneighbors(centers, 1)[-1])
    return np.sort(X[indexes_prototypes])
