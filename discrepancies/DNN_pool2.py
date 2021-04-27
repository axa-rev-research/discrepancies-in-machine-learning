import sys
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()






class BiasedClassifier(privileged_groups, underprivileged_groups, sess):
    def __init__(self):
        self = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='plain_classifier',
                          debias=False, sess=sess)
    
    def fit(self, X, y):
        # transformation de X, y, Z?, en dataset
    
    



class ZhangAIFPool:
    def __init__(self):
        pass
    
    """
    Cette classe va juste servir de wrapper à AIF360. L'intérêt va être de normaliser à l'utilisation de pool.
    Cela inclut
    - faire des classes pour les classifieurs "biased" et "fair", où on doit inclure dans les fonctions fit et predict une transformation des formats de datasets utilisés par nous à ceux par AIF360
    - faire une classe pour la pool pour appeler ça direct
    - dans la pool et dans les classes des clf, permettre de sortir des métriques de fairness (et de visualiser?)
    
    En fait pas sûr, car on va avoir des problèmes rapport aux features catégorielles qui marchent pas dans pool
    Possibilité: passer au génératif direct pour qu'il gère les catégorielles.. S'il l.
    
    """
    
    
    
    
    
    