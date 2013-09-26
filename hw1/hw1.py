import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
#from __future__ import division                                                                                         
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)

# split datasets into n=3 sets                                                               
# set up lists for all possible situations - will run 
# all permutations later in loop.

np.random.seed(0)
indices = np.random.permutation(len(iris_X)) #len(iris_X) = 150  

iris_X_train = []                                                        
iris_X_train.append(iris_X[indices[1:50]])
iris_X_train.append(iris_X[indices[51:100]])
iris_X_train.append(iris_X[indices[101:150]])

iris_y_train = []
iris_y_train.append(iris_y[indices[1:50]])
iris_y_train.append(iris_y[indices[51:100]])
iris_y_train.append(iris_y[indices[101:150]])

iris_X_test = iris_X_train
iris_y_test = iris_y_train

#print iris_X_train[0]

#run knn classifier for all combinations of test/train sets
train_loop = [ [1,2],[0,2],[0,1] ]
test_loop = [0,1,2]
prediction = []
k_value = 5
#print iris_y_train                                                                                                      
for i in test_loop:
   for j in train_loop[i]:
       knn = KNeighborsClassifier(n_neighbors = k_value)  
       knn.fit(iris_X_train[j], iris_y_train[j])
       prediction.append(knn.predict(iris_X_test[i]))

print prediction

#now get accuracy
def accuracy(prediction, iris_y_test):
    diff_count = 0
    for i, j in zip(prediction, iris_y_test):
        if i != j:
            diff_count +=1
            return diff_count
    
knn_score = []    
for n in test_loop:
   for k in train_loop[n]:
      print "Only " + str(accuracy(prediction[k], iris_y_test[n])) + " data point was classified incorrectly in test set "+ str(n)+" train set " + str(k)
      print "In other words, there was " + str(int((1 - accuracy(prediction[k], iris_y_test[n])/len(prediction[k]))*100))+"%" + " accuracy"
      knn_score.append(knn.score(iris_X_test[n], iris_y_test[n]))

print knn_score

print "average generalization error = " + str(np.mean(knn_score))


