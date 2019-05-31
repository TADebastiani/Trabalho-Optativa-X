import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression

data = ds.load_wine()

feature_names = data.feature_names
x = data.data
y = data.target

def KBest(func):
    print("Antes:")
    print(data.feature_names[:-1])
    print(x.shape)

    skb = SelectKBest(score_func=func, k=3)
    best = skb.fit_transform(x, y)

    print("Depois:")
    print([data.feature_names[i] for i in skb.get_support(indices=True)])
    print(best.shape)

    return skb.scores_, skb.pvalues_


chi_scores, chi_pvalues = KBest(chi2)

classif_scores, classif_pvalues = KBest(f_classif)

regression_scores, regression_pvalues = KBest(f_regression)

ind = np.arange(len(feature_names))
width = 0.25

plt.figure(figsize=[10, 10])

plt.bar(ind - width, chi_scores, width, label='chi2')
plt.bar(ind, classif_scores, width, label='f_classif')
plt.bar(ind + width, regression_scores, width, label='f_regression')

plt.title("Scores")
plt.ylabel('Feature')
plt.xticks(ind, feature_names, rotation=90)
plt.ylabel('Pontuação')
plt.legend()

plt.show()
