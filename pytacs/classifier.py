from sklearn.svm import SVC as _SVC
from scanpy import AnnData as _AnnData
import numpy as _np
from scipy.sparse import csr_matrix as _csr_matrix

from .utils import _UNDEFINED
from .utils import subCountMatrix_genes2InGenes1 as _get_subCountMatrix

# >>> ---- Local Classifier ----
class LocalClassifier(_SVC):
    """Based on sklearn.svm.SVC (See relevant reference there),
     specially built for snRNA-seq data training.
    This classifier would predict probabilities for each class, as well as
     the negative-control class (the last class).
    An OVR (One-versus-Rest) strategy is used.

    .fit(), .predict(), and .predict_proba() are specially built, but
     often the last two methods are not to be called manually.

    Predicted integer i indicates the class self.__classes[i].
    
    Parameters
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
        **kwargs
    ):
        self.__threshold_confidence = threshold_confidence
        self.__has_negative_control = False
        self.__genes = _UNDEFINED
        self.__classes = _UNDEFINED
        return super().__init__(
            C=C,
            tol=tol,
            cache_size=cache_size,
            random_state=random_state,
            probability=True,
            **kwargs
        )
    @property
    def threshold_confidence(self) -> float:
        return self.__threshold_confidence
    def set_threshold_confidence(self, value: float = 0.75):
        self.__threshold_confidence = value
        return self
    @property
    def has_negative_control(self) -> bool:
        return self.__has_negative_control
    @property
    def genes(self):
        return self.__genes.copy()
    @property
    def classes(self):
        return self.__classes.copy()

    def classId_to_className(self, class_id: int) -> str:
        """Returns self.__classes[class_id]."""
        return self.__classes[class_id]
    def className_to_classId(self, class_name: str) -> int:
        """Returns the index where `class_name` is in self.__classes."""
        return _np.where(self.__classes==class_name)[0][0]
    def classIds_to_classNames(self, class_ids: _np.ndarray[int]) -> _np.ndarray[str]:
        return _np.array(
            list(
                map(
                    lambda cid: self.classId_to_className(cid),
                    class_ids
                )
            )
        )
    def classNames_to_classIds(self, class_names: _np.ndarray[str]) -> _np.ndarray[int]:
        return _np.array(
            list(
                map(
                    lambda cnm: self.className_to_classId(cnm),
                    class_names
                )
            )
        )

    def fit(
        self,
        sn_adata: _AnnData,
        colname_classes: str = 'cell_type',
    ):
        """Trains the local classifier using the AnnData (h5ad) format snRNA-seq data.
        
        Args:
            sn_adata (AnnData): snRNA-seq h5ad data. Must have attributes:
             .X, the sample-by-gene count matrix;
             .obs['cell_type'] or named otherwise, that indicates the label of each sample;
             .var, whose index indicates the genes used for training.
            
            colname_classes (str): the name of the column in .obs that indicates the
             cell types (classes).
        """
        self.__genes = _np.array(sn_adata.var.index)
        self.__classes = _np.array((sn_adata.obs[colname_classes]).unique())
        # Move the NegativeControl label to the end
        if '__NegativeControl' in self.__classes:
            self.__has_negative_control = True
            self.__classes = _np.concatenate([
                self.__classes[self.__classes!='__NegativeControl'],
                _np.array(['__NegativeControl']),
            ])
        else:
            self.__has_negative_control = False
        # Prepare y: convert classNames into classIds
        class_ids = self.classNames_to_classIds(_np.array(sn_adata.obs[colname_classes]))
        X_train = sn_adata.X
        if type(X_train) is _csr_matrix:
            X_train = X_train.toarray()
        return super().fit(X=X_train, y=class_ids)

    def predict_proba(
        self,
        X: _np.ndarray,
        genes: list[str] | _np.ndarray[str] = None,
    ) -> _np.ndarray[float]:
        """Predicts the probabilities (using Platt normalization) for each sample
         to be of each class."""
        if type(X) is _csr_matrix:
            X = X.toarray()
        assert type(X) is _np.ndarray
        assert (len(X.shape) == 2), 'X must be a sample-by-gene matrix'
        if genes is None:
            genes = self.__genes
        assert (len(genes) == X.shape[1]), 'genes must be compatible with X.shape[1]'
        # Select those genes that appear in self.__genes
        X_new = _get_subCountMatrix(X, self.__genes, genes)
        return super().predict_proba(X_new)

    def predict(
        self,
        X: _np.ndarray,
        genes: list[str] | _np.ndarray[str] = None,
    ) -> _np.ndarray[int]:
        """Predicts classes of each sample. For example, if prediction is i, then
         the predicted class is self.classes[i]. For those below confidence threshold,
         predicted classes are set to -1."""
        probas = self.predict_proba(X, genes)
        classes_pred = _np.argmax(probas, axis=1) # Not considering confidence.
        probas_max = probas[_np.arange(probas.shape[0]), classes_pred]
        ixBool_notConfident = (probas_max < self.threshold_confidence)
        classes_pred[ixBool_notConfident] = -1
        return classes_pred
# ---- Local Classifier ---- <<<