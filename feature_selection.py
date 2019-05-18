import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def UnivariateSelection():
	
	#Select the best features
	bestFeatures = SelectKBest(score_func=chi2, k='all')
	fit = bestFeatures.fit(data.data, data.target)

	# Concat dataframes to print
	dfScores = pd.DataFrame(fit.scores_)
	dfColumns = pd.DataFrame(data.feature_names)
	featureScores = pd.concat([dfColumns, dfScores], axis=1)
	featureScores.columns = ['Specs', 'Score']

	#printing the Top10
	print(featureScores.nlargest(10, 'Score'))

def FeatureImportance():
	data = ds.load_iris()
	
	model = ExtraTreesClassifier(n_estimators=10)
	model.fit(data.data,data.target)
	print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

	#plot graph of feature importances for better visualization
	feat_importances = pd.Series(model.feature_importances_, index=data.feature_names)
	feat_importances.nlargest(10).plot(kind='barh')
	plt.show()

def CorrelationSelection():
	data = ds.load_iris()

	irisData = np.append(data.data, data.target.reshape(data.target.shape[0], 1), axis=1)

	data = pd.DataFrame(irisData, columns=np.append(data.feature_names, ['class']))

	corrmat = data.corr()
	top_corr_features = corrmat.index
	
	plt.figure()

	g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")

	plt.show()

data = ds.load_iris()

print("Univariate Selection")
UnivariateSelection()
print()
print("Feature Importance")
FeatureImportance()
print()
print("Correlation Selection")
CorrelationSelection()
print()