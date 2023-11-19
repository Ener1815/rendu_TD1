from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline


def make_model(task):
    if task == "is_comic_video":
        return RandomForestClassifier()
    elif task == "is_name":
        return Pipeline([
            ("vectorizer", DictVectorizer(sparse=False)),
            ("classifier", RandomForestClassifier(n_estimators =300,criterion = "entropy")),
        ])
    #'gini', 'entropy', 'log_loss'