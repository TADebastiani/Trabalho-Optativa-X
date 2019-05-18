from sklearn import datasets as ds
from sklearn import tree

def Classification():
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(data.data, data.target)
	print(clf)
	# tree.plot_tree(clf)


data = ds.load_iris()

Classification()