import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df  = pd.read_csv("D:\Artificial Intelligence\Dimensionality Reduction\mnist_train.csv")

X = df.iloc[:,1:]
y = df["label"]

X = np.array(X.iloc[:50000,:])
y = np.array(y.iloc[:50000])

#lets split the data

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Now first is to reduce the dimensionality
pca  = PCA(n_components=100)
x_train_reduce = pca.fit_transform(x_train)
x_test_reduce  = pca.fit_transform(x_test)

tsne = TSNE(n_components=2)
x_train_optimize = tsne.fit_transform(x_train_reduce)
x_test_optimize =  tsne.fit_transform(x_test_reduce)




#Now using k-fold cross validation to find the best K

neigbour = np.arange(1,50,2)
total_score = []
for i in neigbour:
    knn  = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn,x_train_optimize,y_train,cv=10,scoring="accuracy")
    total_score.append(scores.mean())


# calculating the miss classification error :

MSE  = [1-x for x in total_score]
optimalK  =  neigbour[MSE.index(min(MSE))]

# Now use the test set to predic the outcome :

knn =  KNeighborsClassifier(n_neighbors=optimalK,algorithm="kd_tree")
knn.fit(x_train_optimize,y_train)
predic  = knn.predict(x_test_optimize)
predic_probability  =  knn.predict_proba(x_test_optimize)
accur =  accuracy_score(y_test,predic,normalize=True)
print(accur)
print(predic_probability)
