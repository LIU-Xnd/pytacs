from sklearn.svm import SVC as _SVC

from sklearn.naive_bayes import GaussianNB as _GaussianNB
from sklearn.multioutput import MultiOutputClassifier as _MultiOutputClassifier

from scanpy import AnnData as _AnnData
import numpy as _np
from scipy.stats import norm as _norm
from scipy.sparse import csr_matrix as _csr_matrix
from typing import Iterable, Literal
from .utils import _UNDEFINED, _Undefined
from .utils import subCountMatrix_genes2InGenes1 as _get_subCountMatrix


# >>> ---- Local Classifier ----
class _LocalClassifier:
    """This classifier would predict probabilities for each class, as well as
     the negative-control class (the last class), if generated.

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
        self._has_negative_control: bool = False
        self._genes: _np.ndarray[str] | _Undefined = _UNDEFINED
        self._classes: _np.ndarray[str] | _Undefined = _UNDEFINED
        return None

    @property
    def threshold_confidence(self) -> float:
        return self._threshold_confidence

    def set_threshold_confidence(self, value: float = 0.75):
        self._threshold_confidence = value
        return self

    @property
    def has_negative_control(self) -> bool:
        return self._has_negative_control

    @property
    def genes(self) -> _np.ndarray[str] | _Undefined:
        return self._genes.copy()

    @property
    def classes(self) -> _np.ndarray[str] | _Undefined:
        return self._classes.copy()

    def classId_to_className(self, class_id: int) -> str:
        """Returns self._classes[class_id]."""
        return self._classes[class_id]

    def className_to_classId(self, class_name: str) -> int:
        """Returns the index where `class_name` is in self._classes."""
        return _np.where(self._classes == class_name)[0][0]

    def classIds_to_classNames(self, class_ids: Iterable[int]) -> _np.ndarray[str]:
        return _np.array(
            list(map(lambda cid: self.classId_to_className(cid), class_ids))
        )

    def classNames_to_classIds(self, class_names: Iterable[str]) -> _np.ndarray[int]:
        return _np.array(
            list(map(lambda cnm: self.className_to_classId(cnm), class_names))
        )

    def fit(
        self,
        sn_adata: _AnnData,
        colname_classes: str = "cell_type",
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
             Negative controls should be named '__NegativeControl' which is the
             default name of negative controls generated
             by pytacs.data.AnnDataPreparer.

        Return:
            dict(X: 2darray, y: array): data ready to train.

        Return (overwritten):
            self (Model): a trained model (self).
        """
        self._genes: _np.ndarray[str] = _np.array(sn_adata.var.index)
        self._classes: _np.ndarray[str] = _np.array(
            (sn_adata.obs[colname_classes]).unique()
        ).astype(str)
        # Move the __NegativeControl label to the end
        if "__NegativeControl" in self._classes:
            self._has_negative_control: bool = True
            self._classes: _np.ndarray[str] = _np.concatenate(
                [
                    self._classes[self._classes != "__NegativeControl"],
                    _np.array(["__NegativeControl"]),
                ]
            )
        else:
            self._has_negative_control: bool = False
        # Prepare y: convert classNames into classIds
        class_ids: _np.ndarray[int] = self.classNames_to_classIds(
            _np.array(sn_adata.obs[colname_classes]).astype(str)
        )
        X_train: _np.ndarray | _csr_matrix = sn_adata.X
        if type(X_train) is _csr_matrix:
            X_train: _np.ndarray[float | int] = X_train.toarray()

        return dict(X=X_train, y=class_ids)

    def predict_proba(
        self,
        X: _np.ndarray | _csr_matrix,
        genes: Iterable[str] | None = None,
    ) -> dict:
        """Predicts the probabilities for each
        sample to be of each class.

        Args:
            X (_np.ndarray | _csr_matrix): input count matrix.

            genes (Iterable[str] | None): list of genes corresponding to
             X's columns. If None, set to pretrained snRNA-seq's gene list.

        Return:
            dict(X: 2darray): X ready to be predictors.

        Return (overwritten):
            2darray[float]: probs of falling into each class;
             each row is a sample and each column is a class.

        Needs overwriting."""

        if type(X) is _csr_matrix:
            X = X.toarray()
        assert type(X) is _np.ndarray
        assert len(X.shape) == 2, "X must be a sample-by-gene matrix"
        if genes is None:
            genes = self._genes
        assert len(
            genes) == X.shape[1], "genes must be compatible with X.shape[1]"
        # Select those genes that appear in self._genes
        X_new = _get_subCountMatrix(X, self._genes, genes)
        # print(f"{X_new.shape=}")
        return {'X': X_new}

    def predict(
        self,
        X: _np.ndarray,
        genes: Iterable[str] | None = None,
    ) -> _np.ndarray[int]:
        """Predicts classes of each sample. For example, if prediction is i,
         then the predicted class is self.classes[i].
         For those below confidence threshold,
         predicted classes are set to -1.

        Args:
            X (_np.ndarray | _csr_matrix): input count matrix.

            genes (Iterable[str] | None): list of genes corresponding to
             those of X's columns. If None, set to pretrained gene list.

        Return:
            _np.ndarray[int]: an array of predicted classIds.

        Needs overwriting."""

        probas: _np.ndarray = self.predict_proba(X, genes)
        assert type(probas) is not dict, ".predict_proba() needs overwriting!"
        # Not considering negative controls.
        classes_pred = _np.argmax(probas, axis=1)
        probas_max = probas[_np.arange(probas.shape[0]), classes_pred]
        where_notConfident = probas_max < self.threshold_confidence
        classes_pred[where_notConfident] = -1
        return classes_pred

# ---- Local Classifier ---- <<<


class SVM(_LocalClassifier):
    """Based on sklearn.svm.SVC (See relevant reference there),
     specially built for snRNA-seq data training.
    This classifier would predict probabilities for each class, as well as
     the negative-control class (the last class).
    An OVR (One-versus-Rest) strategy is used.

    .fit(), .predict(), and .predict_proba() are specially built, but
     often the last two methods are not to be called manually.

    Predicted integer i indicates the class self._classes[i].

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

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`."""

    def __init__(
        self,
        threshold_confidence: float = 0.75,
        C: float = 1.0,
        tol: float = 1e-3,
        cache_size: float = 200,
        random_state: int | None = None,
        **kwargs,
    ):
        self._model = _SVC(
            C=C,
            tol=tol,
            cache_size=cache_size,
            random_state=random_state,
            probability=True,
            **kwargs,
        )
        return super().__init__(threshold_confidence=threshold_confidence)

    def fit(
        self,
        sn_adata: _AnnData,
        colname_classes: str = "cell_type",
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
             indicates the cell types (classes). Negative controls (if exist)
             should be named "__NegativeControl".

        Return:
            self (model)."""

        X_y_ready: dict = super().fit(
            sn_adata=sn_adata,
            colname_classes=colname_classes,
        )

        self._model.fit(X=X_y_ready["X"], y=X_y_ready["y"])
        return self

    def predict_proba(
        self,
        X: _np.ndarray,
        genes: Iterable[str] | None = None,
    ) -> _np.ndarray[float]:
        """Predicts the probabilities for each
         sample to be of each class.

        Args:
            X (_np.ndarray | _csr_matrix): input count matrix.

            genes (Iterable[str] | None): list of genes corresponding to
             X's columns. If None, set to pretrained snRNA-seq's gene list.

        Return:
            2darray[float]: probs of falling into each class;
             each row is a sample and each column is a class."""

        X_ready: dict = super().predict_proba(X, genes)
        return self._model.predict_proba(X_ready["X"])

    def predict(
        self,
        X: _np.ndarray,
        genes: Iterable[str] | None = None,
    ) -> _np.ndarray[int]:
        """Predicts classes of each sample. For example, if prediction is i,
         then the predicted class is self.classes[i].
         For those below confidence threshold,
         predicted classes are set to -1.

        Args:
            X (_np.ndarray | _csr_matrix): input count matrix.

            genes (Iterable[str] | None): list of genes corresponding to
             those of X's columns. If None, set to pretrained gene list.

        Return:
            _np.ndarray[int]: an array of predicted classIds."""

        return super().predict(X, genes)

class GaussianNaiveBayes(_LocalClassifier):
    """This classifier based on Gaussian Naive Bayes models would predict
     probabilities for each class, as well as
     the negative-control class (the last class, but not necessary),
     if generated.

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

    prob_mode: Literal['relative', 'multiplied', 'average'], default='average'
        Three different ways to calculate confidence (probability).
        - 'relative': relative probs among classes;
        - 'multiplied': two-tail cumulative multiplied probs;
        (above two are unstable to outlier features)
        - 'average': average two-tail cumulative probs among features
         (recommended).
    """

    def __init__(self, threshold_confidence: float = 0.75,
                 log1p: bool = True,
                 normalize: bool = False,
                 on_PCs: bool = True,
                 n_PCs: int = 10,
                 prob_mode: Literal['relative', 'multiplied', 'average'] = 'average',
                 **kwargs):
        super().__init__(threshold_confidence=threshold_confidence)
        self._model = _GaussianNB(**kwargs)
        self._normalize: bool = normalize
        self._log1p: bool = log1p
        self._PC_loadings: _np.ndarray | _Undefined = _UNDEFINED
        self._n_PCs: int = n_PCs if on_PCs else 0
        self._on_PCs: bool = on_PCs
        self._prob_mode: str = prob_mode
        assert prob_mode in ['relative', 'multiplied', 'average']
        return None

    def _gaussian_tail_probability(x_obs: float, mean: float, var: float) -> float:
        """Calculate the two-tail probability for
        each feature (assuming Gaussian distribution)"""
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
             Negative controls should be named '__NegativeControl' which is the
             default name of negative controls generated
             by pytacs.data.AnnDataPreparer.

        Return:
            self (Model): a trained model (self).
        """
        X_y_ready: dict = super().fit(
            sn_adata=sn_adata, colname_classes=colname_classes
        )
        X_ready: _np.ndarray = X_y_ready['X']
        if self._normalize:
            X_ready = 1e4 * _np.divide(
                X_ready,
                _np.sum(X_ready, axis=1).reshape(-1,1)
                )
        if self._log1p:
            X_ready = _np.log1p(X_ready)
        if self._on_PCs:
            # Non-centered PCA
            self._PC_loadings = _np.real(  # in case it is complex, but barely
                _np.linalg.svd(a=X_y_ready['X'],
                        full_matrices=False,
                        compute_uv=True,
                        hermitian=False).Vh  # the loading matrix, PC by gene
                        )[:self._n_PCs, :]
            X_ready = X_ready @ self._PC_loadings.T
        self._model.fit(X=X_ready, y=X_y_ready["y"])
        return self

    def predict_proba(
        self, X: _np.ndarray | _csr_matrix, genes: Iterable[str] | None = None
    ) -> _np.ndarray[float]:
        """Predicts the tail probabilities for each
        sample to be of each class.

        Args:
            X (_np.ndarray | _csr_matrix): input count matrix.

            genes (Iterable[str] | None): list of genes corresponding to
             X's columns. If None, set to pretrained snRNA-seq's gene list.

        Return:
            2darray[float]: probs of falling into each class;
             each row is a sample and each column is a class."""
        X_ready: _np.ndarray = super().predict_proba(X, genes)["X"]
        if self._normalize:
            X_ready = 1e4 * _np.divide(
                X_ready,
                _np.sum(X_ready, axis=1).reshape(-1,1)
                )
        if self._log1p:
            X_ready = _np.log1p(X_ready)
        if self._on_PCs:
            X_ready = X_ready @ self._PC_loadings.T

        if self._prob_mode == 'relative':
            return self._model.predict_proba(X_ready)
        
        tail_probabilities = _np.zeros(
            shape=(X_ready.shape[0], self._model.classes_.shape[0])
        )
        for i, sample in enumerate(X_ready):
            for j, class_id in enumerate(self._model.classes_):
                tail_probs = [
                    GaussianNaiveBayes._gaussian_tail_probability(
                        x_obs=sample[k],
                        mean=self._model.theta_[j, k],
                        var=self._model.var_[j, k]
                    ) for k in range(X_ready.shape[1])
                ]  # for each feature.
                if self._prob_mode == 'multiplied':
                    tail_probabilities[i, j] = _np.prod(
                        tail_probs
                    )  # Assuming independence across features
                else:  # average
                    tail_probabilities[i, j] = _np.mean(
                        tail_probs
                    )
        return tail_probabilities

    def predict(
        self, X: _np.ndarray | _csr_matrix, genes: Iterable[str] | None = None
    ) -> _np.ndarray[int]:
        """Predicts classes of each sample. For example, if prediction is i,
         then the predicted class is self.classes[i].
         For those below confidence threshold,
         predicted classes are set to -1.

        Args:
            X (_np.ndarray | _csr_matrix): input count matrix.

            genes (Iterable[str] | None): list of genes corresponding to
             those of X's columns. If None, set to pretrained gene list.

        Return:
            _np.ndarray[int]: an array of predicted classIds."""

        return super().predict(X, genes)

