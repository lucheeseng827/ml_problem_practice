from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import sklearn
from sklearn.datasets import make_friedman1

X, y = make_friedman1(n_samples=500, n_features=15, noise=0.3, random_state=23)

# print(X.shape)
# for run = 1:5
#     (Xtrn, ytrn), (Xtst, ytst) = split data(X), labels(y) #into training & test subsets randomly
#     for depth d = 1:10
#         tree[d] = train decision tree of depth d on the
#             training subset(Xtrn, ytrn)
#         train_scores[run, d] = compute R2 score of tree[d] on the

subsets = ShuffleSplit(n_splits=5, test_size=0.33, random_state=23)
model = DecisionTreeRegressor()
trn_scores, tst_scores = validation_curve(
    model,
    X,
    y,
    param_name="max_depth",
    param_range=range(1, 11),
    cv=subsets,
    scoring="r2",
)
mean_train_score = np.mean(trn_scores, axis=1)
mean_test_score = np.mean(tst_scores, axis=1)
