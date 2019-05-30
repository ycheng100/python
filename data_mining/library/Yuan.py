from sklearn.metrics import auc,roc_curve,log_loss,accuracy_score
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
import pandas as pd # creating data frame for regression
from matplotlib import pyplot as plt

def meshGrid(xlim, ylim, n=30):
    """Create a mesh of points to plot in

    Parameters
    ----------
    xlim: range of x data to base x-axis meshgrid on
    ylim: range of y data to base y-axis meshgrid on
    n: number of points in x and y for meshgrid, optional

    Returns
    -------
    XX, YY : ndarray
    """
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], n)
    yy = np.linspace(ylim[0], ylim[1], n)
    XX, YY = np.meshgrid(xx,yy)
    return XX, YY

def decisionContour(ax, clf, XX, YY,validate_features=True, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    XX: x-axis meshgrid ndarray
    YY: y-axis meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    # Flatten array XX and YY then stack them vertically then transpose: 1st column XX and 2nd column YY
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    if (validate_features==False): # XGBoost
        Z = clf.predict(xy,validate_features=False).reshape(XX.shape)
    else: # Other classifier
        Z = clf.predict(xy).reshape(XX.shape)
		
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, alpha=1)	
    return

# Crossvalidation with AUC scoring on predicted probability (between 0 and 1)
def crossvalidation_auc(numFold,clf,X,y,modelType):
# Note clf is changed inside the function as passed by reference
	#folds = KFold(n_splits=numFold)
	folds = StratifiedKFold(n_splits=numFold)
	numRow=y.shape[0]
	pooledTrue=np.zeros(shape=(len(y)))
	pooledPredict=np.zeros(shape=(len(y)))
	foldMetrics=np.zeros(shape=(numFold,3))

	i=0
	if isinstance(X, pd.DataFrame):
		dataX=X.values
	else:
		dataX=X
	dataY=y.values.ravel()
	startIndex=0
	# n fold cross-validation
	for train_index, test_index in folds.split(dataX,dataY): 
		x_train, x_test = dataX[train_index], dataX[test_index]
		y_train, y_test = dataY[train_index], dataY[test_index]
		endIndex=len(y_test)+startIndex
		# training
		fittedModel=clf.fit(x_train, y_train)
		if (modelType==1): # Model can predict_proba 
			predicted_probability=fittedModel.predict_proba(x_test)
			# Probability for class 1
			y_proba=predicted_probability[:,1]
			featureCount=sum(fittedModel.feature_importances_.ravel()!=0)
		elif (modelType==2): # Model has decision_function
			y_proba=fittedModel.decision_function(x_test)
			featureCount=x_test.shape[1]
		else: # Model has predict: linear regression
			y_proba=fittedModel.predict(x_test)
			featureCount=sum(fittedModel.coef_.ravel()!=0)

	pooledPredict[startIndex:endIndex]=y_proba
	pooledTrue[startIndex:endIndex]=y_test
	fpf, tpf, thresholds =roc_curve(y_test, y_proba, pos_label=1)
	accuracy,yPred=accuracyScore(optimalThreshold(fpf,tpf,thresholds),y_test,y_proba)
	foldMetrics[i,:]=[auc(fpf,tpf),featureCount,accuracy]
	# Go to next fold
	i=i+1
	startIndex=endIndex

	y_proba=pooledPredict.ravel()
	y_true=pooledTrue.ravel()
	# Pooled ROC curve      
	fpf, tpf, thresholds = roc_curve(y_true,y_proba , pos_label=1)	
	accuracy,yPred=accuracyScore(optimalThreshold(fpf,tpf,thresholds),y_true,y_proba)

	return auc(fpf,tpf),accuracy,fpf,tpf,thresholds,foldMetrics

# Benchmark models with cross validation
def benchmarkCV(models,X,y,numFold,modelTypes,modelNames):
# Note models are changed inside the function as passed by reference
  numModel=len(models)
  fpfValues=[]
  senValues=[]
  thresholds=[]
  modelMetrics=np.zeros(shape=(numModel,3))
  cvMetrics=np.zeros(shape=(numModel,numFold,3))

  for j in range(0,numModel):
      clf=models[j]
      modelType=modelTypes[j]
      pooledMetrics,fpf,sen,thre,foldMetrics=crossvalidation_auc(numFold,clf,X,y,modelType)
      modelMetrics[j,:]=pooledMetrics
      cvMetrics[j,:]=foldMetrics
      fpfValues.append(fpf)
      senValues.append(sen)
      thresholds.append(thre)
      print(str(j)+'. '+modelNames[j]+': '+str(pooledMetrics))
      print(foldMetrics)

  return modelMetrics,fpfValues,senValues,thresholds,cvMetrics

# Plot ROC curve# Plot of pooled ROC and compute AUC
def plotRoc(fpfValues,senValues,titleString,labelX=0.6,labelY=0.2):
	fig,ax= plt.subplots()
	# X axis is false-positive fraction
	ax.plot(fpfValues,senValues)
	ax.set_xlabel('False Positive Fraction\n(1-Specificity)')
	ax.set_ylabel('True Positive Fraction\n(Sensitivity)')
	ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
	ax.set_title(titleString)
	plt.text(labelX,labelY,('AUC: '+format(auc(fpfValues,senValues), '.5f')),fontsize=12)
	# Make the figure fit
	plt.tight_layout()	
	return fig,ax

# Plot threshold at crossing of sensitivity and specificity curve
def plotThreshold(thresholds,fpfValues,senValues,titleString,labelX=0,labelY=0.3):
	fig,ax= plt.subplots()
	# X axis is false-positive fraction
	f=senValues
	g=1-fpfValues
	x=thresholds
	ax.plot(x,f)
	ax.plot(x,g)
	# Find index of point where two curves cross
	idx = np.argwhere(np.diff(np.sign(g-f))).flatten()
	ax.plot(x[idx], f[idx], 'ro')
	threshold=x[idx]
	
	ax.set_xlabel('Threshold for classification')
	ax.set_ylabel('Sensitivity/Specificity')
	ax.set_title(titleString)
	plt.legend(['Sensitivity','Specificity'])
	plt.grid(True)
	plt.text(labelX,labelY,('Threshold: '+format(threshold[0], '.3f')),fontsize=12)
	# Make the figure fit
	plt.tight_layout()
	return threshold,f[idx],g[idx]
	
def testAuc(clf,modelType,x_test,y_test):
	if (modelType==1): # Model can predict_proba 
		predicted_probability=clf.predict_proba(x_test)
		# Probability for class 1
		y_proba=predicted_probability[:,1]
	elif (modelType==2): # Model has decision_function
		y_proba=clf.decision_function(x_test)
		featureCount=x_test.shape[1]
	else: # Model has predict: linear regression
		y_proba=clf.predict(x_test)
	fpf, tpf, thresholds = roc_curve(y_test, y_proba , pos_label=1)	  
	return fpf, tpf,thresholds

def optimalThreshold(fpf,tpf,thresholds):
	# Find optimal threshold value at which sensitivity curve and specificity curve cross
	spef=1-fpf
	# Find index of point where two curves cross
	idx = np.argwhere(np.diff(np.sign(tpf-spef))).flatten()
	threshold=thresholds[idx]
	return threshold[0]	

# Compute accuracy of binary classification for regression prediction based on threshold input
def accuracyScore(threshold,y_true,y_regression):
	y_predict=(y_regression>threshold)
	#f1Score=f1_score(y_test,y_predict)    
	return accuracy_score(y_true,y_predict),y_predict

# Compute accuracy, loss and class labels from regression prediction
def classificationMetrics(yTrue,yRegression):
	fpf, tpf, thresholds = roc_curve(yTrue,yRegression, pos_label=1)
	threshold=optimalThreshold(fpf,tpf,thresholds)
	accuracy,yPred=accuracyScore(threshold,yTrue,yRegression)
	return accuracy,log_loss(yTrue,yPred),yPred