from .types import (
    _AnnData,
    _NDArray,
    _Iterable,
    _Literal,
    _csr_matrix,
    _dok_matrix,
    _UndefinedType,
    _UNDEFINED,
)
from sklearn.svm import LinearSVC as _SVC
from sklearn.calibration import CalibratedClassifierCV as _CalibratedClassifierCV

from sklearn.naive_bayes import GaussianNB as _GaussianNB

import numpy as _np
from scipy.stats import norm as _norm
from scipy.sparse import issparse as _issparse
from scipy.sparse.linalg import svds as _svds
from scipy.spatial import cKDTree as _cKDTree
from .utils import rearrange_count_matrix as _rearrange_count_matrix
from .utils import to_array as _to_array
from .utils import truncate_top_n as _truncate_top_n


# >>> ---- Local Classifier ----
class _LocalClassifier:
    """This classifier would predict probabilities for each class

    .fit(), .predict(), and .predict_proba() are specially built, but
     often the last two methods are not to be called manually.

    Predicted integer i indicates the class self._classes[i].

    Needs overwriting.

    Args
    ----------
    threshold_confidence : float, default=0.75
        Confidence according to which whether the classifier predicts a sample
        to be in a class."""

    def __init__(
        self,
        threshold_confidence: float = 0.75,
        **kwargs,
    ):
        self._threshold_confidence: float = threshold_confidence
        self._genes: _NDArray[_np.str_] | _UndefinedType = _UNDEFINED
        self._classes: _NDArray[_np.str_] | _UndefinedType = _UNDEFINED
        return None

    @property
    def threshold_confidence(self) -> float:
        return self._threshold_confidence

    def set_threshold_confidence(self, value: float = 0.75):
        self._threshold_confidence = value
        return self

    @property
    def genes(self) -> _NDArray[_np.str_] | _UndefinedType:
        return self._genes.copy()

    @property
    def classes(self) -> _NDArray[_np.str_] | _UndefinedType:
        return self._classes.copy()

    def classId_to_className(self, class_id: int) -> str:
        """Returns self._classes[class_id]."""
        assert isinstance(self._classes, _np.ndarray)
        return self._classes[class_id]

    def className_to_classId(self, class_name: str) -> int:
        """Returns the index where `class_name` is in self._classes."""
        return _np.where(self._classes == class_name)[0][0]

    def classIds_to_classNames(self, class_ids: _Iterable[int]) -> _NDArray[_np.str_]:
        return _np.array(
            list(map(lambda cid: self.classId_to_className(cid), class_ids))
        )

    def classNames_to_classIds(self, class_names: _Iterable[str]) -> _NDArray[_np.int_]:
        return _np.array(
            list(map(lambda cnm: self.className_to_classId(cnm), class_names))
        )

    def fit(
        self,
        sn_adata: _AnnData,
        colname_classes: str = "cell_type",
        to_dense: bool = False,
    ) -> dict:
        """Trains the local classifier using the AnnData (h5ad) format
        snRNA-seq data.

        Args:
            sn_adata (AnnData): snRNA-seq h5ad data. Must have attributes:
             .X, the sample-by-gene count matrix;
             .obs['cell_type'] or named otherwise, that indicates the label of
             each sample;
             .var, whose index indicates the genes used for training.

            colname_classes (str): the name of the column in .obs that
             indicates the cell types (classes).

        Return:
            dict(X: 2darray, y: array): data ready to train.

        Return (overwritten):
            self (Model): a trained model (self).
        """
        assert _np.all(
            sn_adata.obs.index.astype(_np.int_) == _np.arange(sn_adata.shape[0])
        ), "sn_adata needs tidying using AnnDataPreparer!"
        self._genes = _np.array(sn_adata.var.index)
        if self._genes.shape[0] > 10_000:
            print(
                f"Warning: genes exceed 10,000, might encounter memory issue. You might want to filter genes first."
            )
        self._classes = _np.sort(
            _np.array((sn_adata.obs[colname_classes]).unique()).astype(str)
        )  # sorted alphabetically
        # Prepare y: convert classNames into classIds
        class_ids: _NDArray[_np.int_] = self.classNames_to_classIds(
            _np.array(sn_adata.obs[colname_classes]).astype(str)
        )
        X_train: _NDArray[_np.float_ | _np.int_] | _csr_matrix = sn_adata.X.copy()
        if type(X_train) is _csr_matrix:
            if to_dense:
                X_train = X_train.toarray()

        return dict(X=X_train, y=class_ids)

    def predict_proba(
        self,
        X: _NDArray | _csr_matrix,
        genes: _Iterable[str] | None = None,
        to_dense: bool = False,
    ) -> dict:
        """Predicts the probabilities for each
        sample to be of each class.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             X's columns. If None, set to pretrained snRNA-seq's gene list.

        Return:
            dict(X: 2darray): X ready to be predictors.

        Return (overwritten):
            2darray[float]: probs of falling into each class;
             each row is a sample and each column is a class.

        Needs overwriting."""

        # X: _np.ndarray = _to_array(X)  # <- this needs modifying for mem
        assert len(X.shape) == 2, "X must be a sample-by-gene matrix"
        assert isinstance(self._genes, _Iterable)
        genes_: list[str] = []
        if genes is None:
            genes_ = list(self._genes)
        else:
            assert isinstance(genes, _Iterable)
        genes_ = _np.array(genes)
        assert len(genes_) == X.shape[1], "genes must be compatible with X.shape[1]"
        # Select those genes that appear in self._genes
        X_new: _csr_matrix = _rearrange_count_matrix(X, genes_, _np.array(self._genes))
        if to_dense:
            X_new = X_new.toarray()
        return {"X": X_new}

    def predict(
        self,
        X: _csr_matrix | _NDArray,
        genes: _Iterable[str] | None = None,
    ) -> _NDArray[_np.int_]:
        """Predicts classes of each sample. For example, if prediction is i,
         then the predicted class is self.classes[i].
         For those below confidence threshold,
         predicted classes are set to -1.
        You can set .threshold_confidence by
        .set_threshold_confidence().

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             those of X's columns. If None, set to pretrained gene list.

        Return:
            _NDArray[_np.int_]: an array of predicted classIds.

        Needs overwriting."""

        probas: _NDArray | dict[str, _np.ndarray] = self.predict_proba(X, genes)
        assert isinstance(probas, _np.ndarray), ".predict_proba() needs overwriting!"
        classes_pred = _np.argmax(probas, axis=1)
        probas_max = probas[_np.arange(probas.shape[0]), classes_pred]
        where_notConfident = probas_max < self.threshold_confidence
        classes_pred[where_notConfident] = -1
        return classes_pred


# ---- Local Classifier ---- <<<


class SVM(_LocalClassifier):
    """Based on sklearn.svm.linearSVC (See relevant reference there),
     specially built for snRNA-seq data training.
    This classifier would predict probabilities for each class
    An OVR (One-versus-Rest) strategy is used.

    .fit(), .predict(), and .predict_proba() are specially built, but
     often the last two methods are not to be called manually.

    Predicted integer i indicates the class self._classes[i].

    Note: this is the same model as in K. Benjamin's TopACT (
    https://gitlab.com/kfbenjamin/topact) classifier SVCClassifier.

    Args
    ----------
    threshold_confidence : float, default=0.75
        Confidence according to which whether the classifier predicts a sample
        to be in a class.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty. For an intuitive visualization of the effects
        of scaling the regularization parameter C, see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_scale_c.py`.

    tol : float, default=1e-4
        Tolerance for stopping criterion.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`."""

    def __init__(
        self,
        threshold_confidence: float = 0.75,
        log1p: bool = True,
        normalize: bool = True,
        on_PCs: bool = False,
        n_PCs: int = 10,
        C: float = 1.0,
        tol: float = 1e-4,
        random_state: int | None = None,
        **kwargs,
    ):
        _model = _SVC(
            C=C,
            tol=tol,
            random_state=random_state,
            dual=False,
            **kwargs,
        )
        self._model = _CalibratedClassifierCV(_model)
        self._normalize: bool = normalize
        self._log1p: bool = log1p
        self._PC_loadings: _NDArray | _UndefinedType = _UNDEFINED
        self._n_PCs: int = n_PCs if on_PCs else 0
        self._on_PCs: bool = on_PCs
        return super().__init__(threshold_confidence=threshold_confidence)

    def fit(
        self,
        sn_adata: _AnnData,
        colname_classes: str = "cell_type",
        to_dense: bool = False,
    ):
        """Trains the local classifier using the AnnData (h5ad) format
        snRNA-seq data.

        Args:
            sn_adata (AnnData): snRNA-seq h5ad data. Must have attributes:
             .X, the sample-by-gene count matrix;
             .obs['cell_type'] or named otherwise, that indicates the label
              of each sample;
             .var, whose index indicates the genes used for training.

            colname_classes (str): the name of the column in .obs that
             indicates the cell types (classes).

        Return:
            self (model)."""

        X_y_ready: dict = super().fit(
            sn_adata=sn_adata,
            colname_classes=colname_classes,
            to_dense=to_dense,
        )
        X_ready: _csr_matrix | _NDArray = X_y_ready["X"]
        if self._normalize:
            if _issparse(X_ready):
                # Normalize using sparse-safe operations
                row_sums = X_ready.sum(
                    axis=1
                ).A1  # .A1 to convert the sum to a 1D array
                row_inv = 1.0 / _np.maximum(row_sums, 1e-8)  # Avoid division by zero
                X_ready = _csr_matrix.multiply(X_ready, row_inv[:, _np.newaxis]) * 1e4
            else:
                X_ready = 1e4 * _np.divide(
                    X_ready, _np.maximum(_np.sum(X_ready, axis=1).reshape(-1, 1), 1e-8)
                )
        if self._log1p:
            if _issparse(X_ready):
                X_ready.data = _np.log1p(X_ready.data)
            else:
                X_ready = _np.log1p(X_ready)
        if self._on_PCs:
            if _issparse(X_ready):
                U, S, Vh = _svds(X_ready, k=self._n_PCs)
                self._PC_loadings = _np.real(Vh)
            else:
                # Non-centered PCA
                self._PC_loadings = _np.real(  # in case it is complex, but barely
                    _np.linalg.svd(
                        a=X_ready, full_matrices=False, compute_uv=True, hermitian=False
                    ).Vh  # the loading matrix, PC by gene
                )[: self._n_PCs, :]
            X_ready = X_ready @ self._PC_loadings.T
        self._model.fit(X=X_ready, y=X_y_ready["y"])
        return self

    def predict_proba(
        self,
        X: _csr_matrix | _NDArray,
        genes: _Iterable[str] | None = None,
    ) -> _NDArray[_np.float_]:
        """Predicts the probabilities for each
         sample to be of each class.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             X's columns. If None, set to pretrained snRNA-seq's gene list.

        Return:
            2darray[float]: probs of falling into each class;
             each row is a sample and each column is a class."""

        X_ready: _csr_matrix | _NDArray = super().predict_proba(X, genes)["X"]
        if self._normalize:
            if _issparse(X_ready):
                # Normalize using sparse-safe operations
                row_sums = X_ready.sum(
                    axis=1
                ).A1  # .A1 to convert the sum to a 1D array
                row_inv = 1.0 / _np.maximum(row_sums, 1e-8)  # Avoid division by zero
                X_ready = _csr_matrix.multiply(X_ready, row_inv[:, _np.newaxis]) * 1e4
            else:
                X_ready = 1e4 * _np.divide(
                    X_ready, _np.maximum(_np.sum(X_ready, axis=1).reshape(-1, 1), 1e-8)
                )
        if self._log1p:
            if _issparse(X_ready):
                X_ready.data = _np.log1p(X_ready.data)
            else:
                X_ready = _np.log1p(X_ready)
        if self._on_PCs:
            assert isinstance(self._PC_loadings, _np.ndarray)
            X_ready = X_ready @ self._PC_loadings.T
        return self._model.predict_proba(X_ready)

    def predict(
        self,
        X: _csr_matrix | _NDArray,
        genes: _Iterable[str] | None = None,
    ) -> _NDArray[_np.int_]:
        """Predicts classes of each sample. For example, if prediction is i,
         then the predicted class is self.classes[i].
         For those below confidence threshold,
         predicted classes are set to -1.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             those of X's columns. If None, set to pretrained gene list.

        Return:
            _NDArray[_np.int_]: an array of predicted classIds."""

        return super().predict(X, genes)


class GaussianNaiveBayes(_LocalClassifier):
    """Not Recommended.

    This classifier based on Gaussian Naive Bayes models would predict
     probabilities for each class.

    .fit(), .predict(), and .predict_proba() are specially built, but
     often the last two methods are not to be called manually.

    Predicted integer i indicates the class self._classes[i].

    For other parameters related to GaussianNB, see reference at
     `sklearn.naive_bayes.GaussianNB`.

    Args
    ----------
    threshold_confidence : float, default=0.75
        Confidence according to which whether the classifier predicts a sample
        to be in a class.

    normalize: bool, default=False
        Will process row normalization first.
        (Not in-place. Will not affect spot integration - it will sum up
        raw counts first, and then process.)

    log1p: bool, default=True
        Will process log(1+x) transform on gene count matrix.
        (Not in-place. Will not affect spot integration - it will sum up
        raw counts first, and then process normalization (if specified), and
        then log1p.)

    on_PCs: bool, default=True
        Uses PCs (transform matrix generated from sn_adata, not zero-centered)
        instead of raw genes as the predictors.

    n_PCs: int, default=10
        Number of PCs to use as predictors. Ignored if `on_PCs` is False.

    prob_mode: _Literal['relative', 'multiplied', 'average'], default='average'
        Three different ways to calculate confidence (probability).
        - 'relative': relative probs among classes;
        - 'multiplied': two-tail cumulative multiplied probs;
        (above two are unstable to outlier features)
        - 'average': average two-tail cumulative probs among features.
    """

    def __init__(
        self,
        threshold_confidence: float = 0.75,
        log1p: bool = True,
        normalize: bool = False,
        on_PCs: bool = True,
        n_PCs: int = 10,
        prob_mode: _Literal["relative", "multiplied", "average"] = "average",
        **kwargs,
    ):
        super().__init__(threshold_confidence=threshold_confidence)
        self._model = _GaussianNB(**kwargs)
        self._normalize: bool = normalize
        self._log1p: bool = log1p
        self._PC_loadings: _NDArray | _UndefinedType = _UNDEFINED
        self._n_PCs: int = n_PCs if on_PCs else 0
        self._on_PCs: bool = on_PCs
        self._prob_mode: str = prob_mode
        assert prob_mode in ["relative", "multiplied", "average"]
        return None

    @staticmethod
    def _gaussian_tail_probability(
        x_obs: _NDArray[_np.float_],
        mean: _NDArray[_np.float_],
        var: _NDArray[_np.float_],
    ) -> _NDArray[_np.float_]:
        """Calculate the two-tail probability for
        each feature (assuming Gaussian distribution).

        Args:
            x_obs: 2d-array of observation-by-feature sample matrix.
            mean: 1d-array of feature mean values.
            var: 1d-array of feature variance values.

        Return:
            A 2d-array of sample-by-feature tail probs.
        """
        assert x_obs.shape[1] == mean.shape[0] == var.shape[0]
        return 1 - _np.abs(2 * _norm.cdf(x_obs, loc=mean, scale=_np.sqrt(var)) - 1)

    def fit(self, sn_adata: _AnnData, colname_classes: str = "cell_type"):
        """Trains the local classifier using the AnnData (h5ad) format
        snRNA-seq data.

        Args:
            sn_adata (AnnData): snRNA-seq h5ad data. Must have attributes:
             .X, the sample-by-gene count matrix;
             .obs['cell_type'] or named otherwise, that indicates the label of
             each sample;
             .var, whose index indicates the genes used for training.

            colname_classes (str): the name of the column in .obs that
             indicates the cell types (classes).

        Return:
            self (Model): a trained model (self).
        """
        X_y_ready: dict = super().fit(
            sn_adata=sn_adata, colname_classes=colname_classes
        )
        X_ready: _NDArray = X_y_ready["X"]
        if self._normalize:
            X_ready = 1e4 * _np.divide(
                X_ready, _np.maximum(_np.sum(X_ready, axis=1).reshape(-1, 1), 1e-8)
            )
        if self._log1p:
            X_ready = _np.log1p(X_ready)
        if self._on_PCs:
            # Non-centered PCA
            self._PC_loadings = _np.real(  # in case it is complex, but barely
                _np.linalg.svd(
                    a=X_ready, full_matrices=False, compute_uv=True, hermitian=False
                ).Vh  # the loading matrix, PC by gene
            )[: self._n_PCs, :]
            X_ready = X_ready @ self._PC_loadings.T
        self._model.fit(X=X_ready, y=X_y_ready["y"])
        return self

    def predict_proba(
        self, X: _NDArray | _csr_matrix, genes: _Iterable[str] | None = None
    ) -> _NDArray[_np.float_]:
        """Predicts the probabilities for each
        sample to be of each class.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             X's columns. If None, set to pretrained snRNA-seq's gene list.

        Return:
            2darray[float]: probs of falling into each class;
             each row is a sample and each column is a class."""
        X_ready: _NDArray = super().predict_proba(X, genes)["X"]
        if self._normalize:
            X_ready = 1e4 * _np.divide(
                X_ready, _np.maximum(_np.sum(X_ready, axis=1).reshape(-1, 1), 1e-8)
            )
        if self._log1p:
            X_ready = _np.log1p(X_ready)
        if self._on_PCs:
            assert isinstance(self._PC_loadings, _np.ndarray)
            X_ready = X_ready @ self._PC_loadings.T

        if self._prob_mode == "relative":
            return self._model.predict_proba(X_ready)
        assert isinstance(self._classes, _np.ndarray)
        tail_probabilities = _np.zeros(shape=(X_ready.shape[0], len(self._classes)))
        for j_cls, _ in enumerate(self._classes):
            tail_probs = GaussianNaiveBayes._gaussian_tail_probability(
                x_obs=X_ready,
                mean=self._model.theta_[j_cls, :],
                var=self._model.var_[j_cls, :],
            )  # Probs of this class
            if self._prob_mode == "multiplied":
                tail_probabilities[:, j_cls] = _np.prod(
                    tail_probs,
                    axis=1,
                )  # Assuming independence across features
            else:  # average
                tail_probabilities[:, j_cls] = _np.mean(tail_probs, axis=1)
        return tail_probabilities

    def predict(
        self, X: _NDArray | _csr_matrix, genes: _Iterable[str] | None = None
    ) -> _NDArray[_np.int_]:
        """Predicts classes of each sample. For example, if prediction is i,
         then the predicted class is self.classes[i].
         For those below confidence threshold,
         predicted classes are set to -1.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             those of X's columns. If None, set to pretrained gene list.

        Return:
            _NDArray[_np.int_]: an array of predicted classIds."""

        return super().predict(X, genes)


class QProximityClassifier(_LocalClassifier):
    """This classifier based on q-proximity confidence metric would predict
     probabilities for each class.

    .fit(), .predict(), and .predict_proba() are specially built, but
     often the last two methods are not to be called manually.

    Predicted integer i indicates the class self._classes[i].

    The proximity radius R(i,q) of class i
    and of quantile q (0 < q < 1) is defined as the minimal r where the ball
    centered at class i's mean vector mu_i with radius r, Ball(mu_i, r),
    contains a proportion of q points in class i. (q<1 to avoid outliers)

    The q-proximity of point x to
    class i, prox_q(x,i), is defined as the cardinality of the intersection
    of Ball(x, R(i,q)) with the set of all points in class i, X_i,
    divided by q * cardinality(X_i), i.e.,
    prox_q(x,i) := cardinality(Ball(x, R(i,q)) & X_i) / (q*cardinality(X_i))

    Args
    ----------
    threshold_confidence : float, default=0.75
        Confidence according to which whether the classifier predicts a sample
        to be in a class.

    normalize: bool, default=False
        Will process row normalization first.
        (Not in-place. Will not affect spot integration - it will sum up
        raw counts first, and then process normalization.)

    log1p: bool, default=True
        Will process log(1+x) transform on gene count matrix.
        (Not in-place. Will not affect spot integration - it will sum up
        raw counts first, and then process normalization (if specified), and
        then log1p.)

    on_PCs: bool, default=True
        Uses PCs (transform matrix generated from sn_adata, not zero-centered)
        instead of raw genes as the predictors.

    n_PCs: int, default=10
        Number of PCs to use as predictors. Ignored if `on_PCs` is False.

    standardize_PCs: bool, default=True
        Standardizes the PCs to make scales of different dimensions of PCs
        consistent. It is for the proximity balls to work better. ignored
        if `on_PCs` is False.

    q: float, default=0.85
        quantile (q), the proportion of points contained by the proximity ball.
        0 < q < 1.

    capped: bool, default=True
        To avoid potential cases where prox_q > 1, we can use
        capped prox_q(x,i) :=
        min(1, prox_q(x,i)) as the confidence metric.
    """

    def __init__(
        self,
        threshold_confidence: float = 0.75,
        log1p: bool = True,
        normalize: bool = False,
        on_PCs: bool = True,
        n_PCs: int = 10,
        standardize_PCs: bool = True,
        q: float = 0.85,
        capped: bool = True,
    ):
        super().__init__(threshold_confidence=threshold_confidence)
        # The ._model_points is a dict mapping class_id to processed sample
        # matrix of that class.
        self._model_points: dict[(int, _NDArray)] = dict()
        # The ._model_radii is a dict mapping class_id to radius of proximity
        # ball of that class.
        self._model_radii: dict[(int, float)] = dict()
        # The ._model_n_points_keep is a dict mapping class_id to number of
        # points to keep for a ball in that class
        self._model_n_points_keep: dict[(int, int)] = dict()

        self._normalize: bool = normalize
        self._log1p: bool = log1p
        self._PC_loadings: _NDArray | _UndefinedType = _UNDEFINED
        self._n_PCs: int = n_PCs if on_PCs else 0
        self._on_PCs: bool = on_PCs
        self._standardize_PCs: bool = standardize_PCs and on_PCs

        # Saves singular values of SVD transform for standardizing PCs.
        self._singular_values: _NDArray[_np.float_] | _UndefinedType = _UNDEFINED
        self._q: float = q

        self._capped: bool = capped
        return None

    def fit(self, sn_adata: _AnnData, colname_classes: str = "cell_type"):
        """Trains the local classifier using the AnnData (h5ad) format
        snRNA-seq data.

        Args:
            sn_adata (AnnData): snRNA-seq h5ad data. Must have attributes:
             .X, the sample-by-gene count matrix;
             .obs['cell_type'] or named otherwise, that indicates the label of
             each sample;
             .var, whose index indicates the genes used for training.

            colname_classes (str): the name of the column in .obs that
             indicates the cell types (classes).

        Return:
            self (Model): a trained model (self).
        """
        X_y_ready: dict = super().fit(
            sn_adata=sn_adata, colname_classes=colname_classes
        )
        X_ready: _NDArray = X_y_ready["X"]

        if self._normalize:
            X_ready = 1e4 * _np.divide(
                X_ready, _np.maximum(_np.sum(X_ready, axis=1).reshape(-1, 1), 1e-8)
            )
        if self._log1p:
            X_ready = _np.log1p(X_ready)
        if self._on_PCs:
            # Non-centered PCA
            svd_results = _np.linalg.svd(
                a=X_ready, full_matrices=False, compute_uv=True, hermitian=False
            )
            self._PC_loadings = _np.real(  # in case it is complex, but barely
                svd_results.Vh  # the loading matrix, PC by gene
            )[: self._n_PCs, :]
            self._singular_values = svd_results.S[: self._n_PCs]

            X_ready = X_ready @ self._PC_loadings.T
            if self._standardize_PCs:
                X_ready = X_ready / self._singular_values

        # "Train" the model
        assert type(self.classes) is _np.ndarray
        for i_cls, cls_name in enumerate(self.classes):
            # Save the points belonging to this class
            self._model_points[i_cls] = X_ready[X_y_ready["y"] == i_cls, :].copy()
            # Find centroid
            n_points: int = self._model_points[i_cls].shape[0]
            mean_vector = self._model_points[i_cls].sum(axis=0) / n_points
            # Calculate distances of points to mean vector
            distances_to_mean = _np.zeros(shape=(n_points,))
            for i_point, point in enumerate(self._model_points[i_cls]):
                distances_to_mean[i_point] = _np.linalg.norm(point - mean_vector)
            # Points from near to far
            ix_from_near_to_far = _np.argsort(distances_to_mean)
            n_points_keep: int = int(_np.round(n_points * self._q))
            n_points_keep = max(n_points_keep, 1)
            self._model_n_points_keep[i_cls] = n_points_keep
            # Point at the ball boundary
            i_support_point: int = ix_from_near_to_far[n_points_keep - 1]
            radius_q: float = distances_to_mean[i_support_point]
            self._model_radii[i_cls] = radius_q
        return self

    def predict_proba(
        self, X: _NDArray | _csr_matrix, genes: _Iterable[str] | None = None
    ) -> _NDArray[_np.float_]:
        """Predicts the probabilities for each
        sample to be of each class.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             X's columns. If None, set to pretrained snRNA-seq's gene list.

        Return:
            2darray[float]: probs of falling into each class;
             each row is a sample and each column is a class."""

        X_ready: _NDArray = super().predict_proba(X, genes)["X"]
        if self._normalize:
            X_ready = 1e4 * _np.divide(
                X_ready, _np.maximum(_np.sum(X_ready, axis=1).reshape(-1, 1), 1e-8)
            )
        if self._log1p:
            X_ready = _np.log1p(X_ready)
        if self._on_PCs:
            assert type(self._PC_loadings) is _np.ndarray
            X_ready = X_ready @ self._PC_loadings.T
        if self._standardize_PCs:
            X_ready = X_ready / self._singular_values
        # Probs
        assert type(self._classes) is _np.ndarray
        probs = _np.zeros(shape=(X_ready.shape[0], len(self._classes)))

        # Compute distances of each sample to each ref points in the class
        # Update 3.4.1: use cKDTree to implement this and parallelize it
        tree_samples = _cKDTree(
            data=X_ready,
        )
        for i_class, _ in enumerate(self._classes):
            tree_refs = _cKDTree(self._model_points[i_class])
            dist_matrix: _dok_matrix = tree_samples.sparse_distance_matrix(
                other=tree_refs,
                max_distance=self._model_radii[i_class],
                p=2,
                output_type="dok_matrix",
            ).astype(
                bool
            )  # True indicates proximal and False vice-versa.

            # Count proximal point proportion for each sample
            n_proximal: _NDArray[_np.int_] = _to_array(
                dist_matrix.sum(axis=1), squeeze=True
            )
            probs[:, i_class] = n_proximal / self._model_n_points_keep[i_class]
            if self._capped:
                probs[:, i_class] = _np.minimum(probs[:, i_class], 1.0)

        return probs

    def predict(
        self, X: _NDArray | _csr_matrix, genes: _Iterable[str] | None = None
    ) -> _NDArray[_np.int_]:
        """Predicts classes of each sample. For example, if prediction is i,
         then the predicted class is self.classes[i].
         For those below confidence threshold,
         predicted classes are set to -1.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             those of X's columns. If None, set to pretrained gene list.

        Return:
            _NDArray[_np.int_]: an array of predicted classIds."""

        return super().predict(X, genes)


class CosineSimilarityClassifier(_LocalClassifier):
    """This classifier would predict probs for each class.

    Ref samples of counts of each class are normalized,
    log1p transformed (if specified), and averaged as the ref signatures.
    Pearson's correlation is used to determined the probs for each sample
    to be in a class.

    .fit(), .predict(), and .predict_proba() are specially built, but
    often the last two methods are not to be called manually.

    Predicted integer i indicates the class self._classes[i].

    Args
    ----------
    threshold_confidence : float, default=0.75
        Confidence according to which whether the classifier predicts a sample
        to be in a class.

    log1p : bool, default=True
        Whether to compare log1p transformed expression vectors instead of
        raw counts.
    """

    def __init__(
        self,
        threshold_confidence: float = 0.75,
        log1p: bool = True,
    ):
        super().__init__(threshold_confidence=threshold_confidence)
        self._log1p = log1p
        self._model_signatures: _np.ndarray | _UndefinedType = _UNDEFINED
        return

    def fit(
        self,
        sn_adata: _AnnData,
        colname_classes: str = "cell_type",
    ) -> dict:
        """Trains the local classifier using the AnnData format snRNA-seq data.

        Args:
            sn_adata (AnnData): snRNA-seq h5ad data. Must have attributes:
             .X, the sample-by-gene count matrix;
             .obs['cell_type'] or named otherwise, that indicates the label of
             each sample;
             .var, whose index indicates the genes used for training.

            colname_classes (str): the name of the column in .obs that
             indicates the cell types (classes).

        Return:
            self (Model): a trained model (self).
        """
        data_ready: dict = super().fit(
            sn_adata=sn_adata,
            colname_classes=colname_classes,
        )

        # "Train" the model
        assert isinstance(self._classes, _np.ndarray)
        self._model_signatures = _np.zeros(
            shape=(len(self._classes), len(self._genes)),
        )
        for i_cls, cls_name in enumerate(self._classes):
            # Normalized
            signature: _np.ndarray = data_ready["X"][data_ready["y"] == i_cls, :]
            signature = 1e4 * _np.divide(
                signature,
                _np.maximum(
                    1e-8,
                    _np.sum(signature, axis=1).reshape(-1, 1),
                ),
            )
            if self._log1p:
                signature = _np.log1p(signature)
            # Average
            signature = signature.mean(axis=0)
            # Standardize
            signature /= max(1e-8, _np.linalg.norm(signature))
            self._model_signatures[i_cls, :] = signature
        return self

    def predict_proba(
        self,
        X: _NDArray | _csr_matrix,
        genes: _Iterable[str] | None = None,
    ) -> _NDArray[_np.float_]:
        """Predicts the probabilities for each
        sample to be of each class.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             X's columns. If None, set to pretrained snRNA-seq's gene list.

        Return:
            2darray[float]: probs of falling into each class;
             each row is a sample and each column is a class."""
        X_ready: _np.ndarray = super().predict_proba(X, genes)["X"]
        if self._log1p:
            X_ready = 1e4 * _np.divide(
                X_ready,
                _np.maximum(1e-8, _np.sum(X_ready, axis=1).reshape(-1, 1)),
            )
            X_ready = _np.log1p(X_ready)
        # Standardize
        X_ready = _np.divide(
            X_ready,
            _np.maximum(
                1e-8,
                _np.linalg.norm(X_ready, axis=1).reshape(-1, 1),
            ),
        )
        corr = _np.maximum(1e-8, X_ready @ self._model_signatures.T)
        return corr

    def predict(
        self,
        X: _np.ndarray,
        genes: _Iterable[str] | None = None,
    ):
        """Predicts classes of each sample. For example, if prediction is i,
         then the predicted class is self.classes[i].
         For those below confidence threshold,
         predicted classes are set to -1.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             those of X's columns. If None, set to pretrained gene list.

        Return:
            _NDArray[_np.int_]: an array of predicted classIds."""
        return super().predict(X, genes)


class JaccardClassifier(_LocalClassifier):
    """Not Recommended.

    This classifier would predict probs for each class.

    Ref samples of counts of each class are normalized,
    log1p transformed (if specified), and averaged as the ref signatures.
    Pearson's correlation is used to determined the probs for each sample
    to be in a class.

    .fit(), .predict(), and .predict_proba() are specially built, but
    often the last two methods are not to be called manually.

    Predicted integer i indicates the class self._classes[i].

    Jaccard metric:
    Jacc(arr1, arr2) = mean(bool_(arr1)==bool_(arr2))

    Args
    ----------
    threshold_confidence : float, default=0.75
        Confidence according to which whether the classifier predicts a sample
        to be in a class.

    log1p : bool, default=True
        Whether to compare log1p transformed expression vectors instead of
        raw counts.
    """

    def __init__(
        self,
        threshold_confidence: float = 0.75,
        log1p: bool = True,
    ):
        super().__init__(threshold_confidence=threshold_confidence)
        self._log1p = log1p
        self._model_signatures: _np.ndarray | _UndefinedType = _UNDEFINED
        return

    def fit(
        self,
        sn_adata: _AnnData,
        colname_classes: str = "cell_type",
        n_top_genes_truncated: int | None = None,
    ) -> dict:
        """Trains the local classifier using the AnnData format snRNA-seq data.

        Args:
            sn_adata (AnnData): snRNA-seq h5ad data. Must have attributes:
             .X, the sample-by-gene count matrix;
             .obs['cell_type'] or named otherwise, that indicates the label of
             each sample;
             .var, whose index indicates the genes used for training.

            colname_classes (str): the name of the column in .obs that
             indicates the cell types (classes).

            n_top_genes_truncated (int | None): only the first `n_top_genes_truncated` (int)
            many expressed genes are kept as True (expressed), the rest are set to False.
            `None` for not truncating.

        Return:
            self (Model): a trained model (self).
        """
        data_ready: dict = super().fit(
            sn_adata=sn_adata,
            colname_classes=colname_classes,
        )

        # "Train" the model
        assert isinstance(self._classes, _np.ndarray)
        self._model_signatures = _np.zeros(
            shape=(len(self._classes), len(self._genes)),
            dtype=bool,
        )
        for i_cls, cls_name in enumerate(self._classes):
            signature: _np.ndarray = data_ready["X"][data_ready["y"] == i_cls, :]
            # Average
            signature = signature.mean(axis=0)
            if n_top_genes_truncated is not None:
                assert isinstance(n_top_genes_truncated, int)
                signature = _truncate_top_n(signature, n_top=n_top_genes_truncated)
            self._model_signatures[i_cls, :] = _np.bool_(signature)
        return self

    def predict_proba(
        self,
        X: _NDArray | _csr_matrix,
        genes: _Iterable[str] | None = None,
    ) -> _NDArray[_np.float_]:
        """Predicts the probabilities for each
        sample to be of each class.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             X's columns. If None, set to pretrained snRNA-seq's gene list.

        Return:
            2darray[float]: probs of falling into each class;
             each row is a sample and each column is a class."""
        X_ready: _np.ndarray = super().predict_proba(X, genes)["X"]
        probas = _np.zeros(
            shape=(X_ready.shape[0], len(self.classes)),
        )
        for i_cls, _ in enumerate(self.classes):
            probas[:, i_cls] = (
                _np.bool_(X_ready) == self._model_signatures[i_cls, :]
            ).mean(axis=1)
        return probas

    def predict(
        self,
        X: _np.ndarray,
        genes: _Iterable[str] | None = None,
    ):
        """Predicts classes of each sample. For example, if prediction is i,
         then the predicted class is self.classes[i].
         For those below confidence threshold,
         predicted classes are set to -1.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             those of X's columns. If None, set to pretrained gene list.

        Return:
            _NDArray[_np.int_]: an array of predicted classIds."""
        return super().predict(X, genes)
