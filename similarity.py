from typing import List, Text

from sklearn.feature_extraction.text import (
    TfidfVectorizer, TfidfTransformer, CountVectorizer
)
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import scipy.sparse as sp


def _add_sparse_column(sparse, column):
    addition = sp.lil_matrix(sparse.shape)
    sparse_coo = sparse.tocoo()
    for i, j, v in zip(sparse_coo.row, sparse_coo.col, sparse_coo.data):
        addition[i, j] = v + column[i, 0]
    return addition.tocsr()


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)


class BM25Transformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a normalized tf or tf-idf representation
    Tf means term-frequency while tf-idf means term-frequency times inverse
    document-frequency. This is a common term weighting scheme in information
    retrieval, that has also found good use in document classification.
    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.
    The actual formula used for tf-idf is tf * (idf + 1) = tf + tf * idf,
    instead of tf * idf. The effect of this is that terms with zero idf, i.e.
    that occur in all documents of a training set, will not be entirely
    ignored. The formulas used to compute tf and idf depend on parameter
    settings that correspond to the SMART notation used in IR, as follows:
    Tf is "n" (natural) by default, "l" (logarithmic) when sublinear_tf=True.
    Idf is "t" when use_idf is given, "n" (none) otherwise.
    Normalization is "c" (cosine) when norm='l2', "n" (none) when norm=None.
    Parameters
    ----------
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    use_idf : boolean, optional
        Enable inverse-document-frequency reweighting.
    smooth_idf : boolean, optional
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, optional
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [Yates2011] `R. Baeza-Yates and B. Ribeiro-Neto (2011). Modern
                   Information Retrieval. Addison Wesley, pp. 68-74.`
    .. [MRS2008] `C.D. Manning, P. Raghavan and H. Schuetze  (2008).
                   Introduction to Information Retrieval. Cambridge University
                   Press, pp. 118-120.`
    """

    def __init__(self, norm='l2', use_idf=True, use_bm25idf=False, smooth_idf=True,
                 sublinear_tf=False, bm25_tf=False, k1=1.2, b=0.75, epsilon=0.01):
        self.norm = norm
        self.use_idf = use_idf
        self.use_bm25idf = use_bm25idf
        self.smooth_idf = smooth_idf

        self.sublinear_tf = sublinear_tf
        self.bm25_tf = bm25_tf
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)
        This algorithm sets a floor on the idf values to eps * average_idf
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)

        if self.use_idf:
            n_samples, n_features = X.shape

            # vanilla bm25 idf
            freq = _document_frequency(X)

            # perform idf smoothing if required
            freq += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log1p instead of log makes sure terms with zero idf don't get
            # suppressed entirely
            bm25idf = np.log((n_samples - freq + 0.5) / (freq + 0.5))

            # collect terms with negative idf to set them a special epsilon value.
            # idf can be negative if word is contained in more than half of documents
            bm25idf[bm25idf < 0] = self.epsilon * bm25idf[bm25idf > 0].mean()

            self._idf_diag = sp.spdiags(bm25idf,
                                        diags=0, m=n_features, n=n_features)

            return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.bm25_tf:

            # First calculate the denominator of BM25 equation
            # Sum the frequencies (sum of each row) to get the documents lengths
            D = (X.sum(1) / np.average(X.sum(1))).reshape((n_samples, 1))
            D = ((1 - self.b) + self.b * D) * self.k1
            # D = sp.csr_matrix(np.multiply(np.ones((n_samples,n_features)),D))
            D_X = _add_sparse_column(X, D)

            # Then perform the main division
            # ...Find a better way to add a numpy ndarray to a sparse matrix
            np.divide(X.data * (self.k1 + 1), D_X.data, X.data)
            # np.divide(X.data * (self.k + 1), sp.csr_matrix(np.add(X.todense(), D)).data, X.data)

        elif self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            if not hasattr(self, "_idf_diag"):
                raise ValueError("idf vector not fitted")
            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def idf_(self):
        if hasattr(self, "_idf_diag"):
            return np.ravel(self._idf_diag.sum(axis=0))
        else:
            return None


class BM25Vectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of TF-IDF features.
    Equivalent to CountVectorizer followed by TfidfTransformer.
    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.
        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.
        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.
    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.
    analyzer : string, {'word', 'char'} or callable
        Whether the feature should be made of word or character n-grams.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.
    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    stop_words : string {'english'}, list, or None (default)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.
    lowercase : boolean, default True
        Convert all characters to lowercase before tokenizing.
    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if `analyzer == 'word'`. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    max_df : float in range [0.0, 1.0] or int, optional, 1.0 by default
        When building the vocabulary ignore terms that have a term frequency
        strictly higher than the given threshold (corpus specific stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    min_df : float in range [0.0, 1.0] or int, optional, 1 by default
        When building the vocabulary ignore terms that have a term frequency
        strictly lower than the given threshold.
        This value is also called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_features : optional, None by default
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None.
    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.
    binary : boolean, False by default.
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs.)
    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    use_idf : boolean, optional
        Enable inverse-document-frequency reweighting.
    smooth_idf : boolean, optional
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, optional
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    Attributes
    ----------
    idf_ : array, shape = [n_features], or None
        The learned idf vector (global term weights)
        when ``use_idf`` is set to True, None otherwise.
    See also
    --------
    CountVectorizer
        Tokenize the documents and count the occurrences of token and return
        them as a sparse matrix
    TfidfTransformer
        Apply Term Frequency Inverse Document Frequency normalization to a
        sparse matrix of occurrence counts.
    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, use_bm25idf=False,
                 smooth_idf=True, sublinear_tf=False, bm25_tf=False):
        super(BM25Vectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self.bm25_tf = bm25_tf
        self.use_bm25idf = use_bm25idf
        self._tfidf = BM25Transformer(norm=norm, use_idf=use_idf,
                                      use_bm25idf=use_bm25idf,
                                      smooth_idf=smooth_idf,
                                      sublinear_tf=sublinear_tf,
                                      bm25_tf=bm25_tf)

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value

    @property
    def idf_(self):
        return self._tfidf.idf_

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.
        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
        Returns
        -------
        self : BM25Vectorizer
        """
        X = super(BM25Vectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return term-document matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        X = super(BM25Vectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X, y)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents, copy=True):
        """Transform documents to document-term matrix.
        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).
        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        X = super(BM25Vectorizer, self).transform(raw_documents)
        return self._tfidf.transform(X, copy=False)


def compute_bm25_similarity(documents: List[Text]) -> np.ndarray:
    count_vect = CountVectorizer(min_df=2, lowercase=False)
    bm25 = BM25Transformer(use_bm25idf=True, bm25_tf=True)

    freqs = count_vect.fit_transform(documents)
    tfidf = bm25.fit_transform(freqs)

    n_samples, n_features = tfidf.shape

    similarity_mat = np.zeros((n_samples, n_samples), dtype=np.float32)

    rows, cols = tfidf.nonzero()

    for query_doc_id in np.unique(rows):
        query_terms = cols[rows == query_doc_id]
        similarity_mat[query_doc_id, :] = tfidf[:, query_terms].sum(axis=1).flatten()

    return similarity_mat
