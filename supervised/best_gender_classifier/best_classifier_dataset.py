import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
import numpy as np

# Import the adult.txt file into Python
data = pd.read_csv('adults.txt', sep=',')

#LabelEncoder takes [0,1] for ['Male','Female']
label_encoder = LabelEncoder().fit(['Male','Female'])

for label in ['race', 'occupation']:
    data[label] = LabelEncoder().fit_transform(data[label])

# Take the fields of interest and plug them into variable X
X = data[['race', 'hours_per_week', 'occupation']]
# Make sure to provide the corresponding truth value
Y = label_encoder.transform(data['sex'])

# Split the data into test and training (80% for test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

# Instantiate the Random_Forest classifier
randF_clf = RandomForestClassifier(n_estimators=1000)

# Instantiate the naive_bayes classifier
naiveB_clf = GaussianNB()

# Instantiate the perceptron classifier
per_clf = Perceptron()

# Instantiate the LogisticRegression classifier
logR_clf = LogisticRegression()

# Instantiate the SVC classifier
SVC_clf = SVC()

# Instantiate the KNeighboursClassifier classifier
KNC_clf =  KNeighborsClassifier()

# Instantiate the DecisionTree classifier
DTC_clf = tree.DecisionTreeClassifier()

# Train the classifier using the train data
randF_clf = randF_clf.fit(X_train, Y_train)
naiveB_clf = naiveB_clf.fit(X_train, Y_train)
per_clf = per_clf.fit(X_train, Y_train)
logR_clf = logR_clf.fit(X_train, Y_train)
SVC_clf = SVC_clf.fit(X_train, Y_train)
KNC_clf = KNC_clf.fit(X_train, Y_train)
DTC_clf = DTC_clf.fit(X_train, Y_train)

# Validate the classifier
randF_acc = str( randF_clf.score(X_test, Y_test) )
naiveB_acc = str(naiveB_clf.score(X_test, Y_test))
per_acc = str(per_clf.score(X_test, Y_test))
logR_acc = str(logR_clf.score(X_test,Y_test))
SVC_acc = str(SVC_clf.score(X_test,Y_test))
KNC_acc = str(KNC_clf.score(X_test,Y_test))
DTC_acc = str(DTC_clf.score(X_test,Y_test))

# Make a Prediction
randF_pred = randF_clf.predict(X_test)
naiveB_pred = naiveB_clf.predict(X_test)
per_pred = per_clf.predict(X_test)
logR_pred = logR_clf.predict(X_test)
SVC_pred = SVC_clf.predict(X_test)
KNC_pred = KNC_clf.predict(X_test)
DTC_pred = DTC_clf.predict(X_test)

# Make f1Score , precisionScore , recallScore
randF_f1scr = f1_score(randF_pred,Y_test)
randF_pscr = precision_score(randF_pred,Y_test)
randF_rscr = recall_score(randF_pred,Y_test)

naiveB_f1scr = f1_score(naiveB_pred,Y_test)
naiveB_pscr = precision_score(naiveB_pred,Y_test)
naiveB_rscr = recall_score(naiveB_pred,Y_test)

per_f1scr = f1_score(per_pred,Y_test)
per_pscr = precision_score(per_pred,Y_test)
per_rscr = recall_score(per_pred,Y_test)

logR_f1scr = f1_score(logR_pred,Y_test)
logR_pscr = precision_score(logR_pred,Y_test)
logR_rscr = recall_score(logR_pred,Y_test)

SVC_f1scr = f1_score(SVC_pred,Y_test)
SVC_pscr = precision_score(SVC_pred,Y_test)
SVC_rscr = recall_score(SVC_pred,Y_test)

KNC_f1scr = f1_score(KNC_pred,Y_test)
KNC_pscr = precision_score(KNC_pred,Y_test)
KNC_rscr = recall_score(KNC_pred,Y_test)

DTC_f1scr = f1_score(DTC_pred,Y_test)
DTC_pscr = precision_score(DTC_pred,Y_test)
DTC_rscr = recall_score(DTC_pred,Y_test)

#Calculating the best classifiers based on score
max_score = np.argmax( [randF_acc ,naiveB_acc ,
per_acc ,logR_acc ,
SVC_acc ,KNC_acc ,DTC_acc] )
classifiers = {
               0:'Random Forest Classifer' ,
               1:'Naive Bayes Classifier',
               2:'Perceptron Classifier',
               3:'Logistic Regression Classifier',
               4:'Support Vector Classifier',
               5:'K Neighbours Classifier',
               6:'Decision Tree Classifier' 
                 }
print( ' Best Classifier by Score => ',classifiers[max_score])




max_pscr =  np.argmax( [randF_pscr ,naiveB_pscr ,
per_pscr ,logR_pscr ,
SVC_pscr ,KNC_pscr ,DTC_pscr] )

print( ' Best Classifier by Precision score => ',classifiers[max_pscr])

max_rscr =  np.argmax( [randF_rscr ,naiveB_rscr ,
per_rscr ,logR_rscr ,
SVC_rscr ,KNC_rscr ,DTC_rscr] )

print( ' Best Classifier by Recall score => ',classifiers[max_rscr])

max_f1scr =  np.argmax( [randF_f1scr ,naiveB_f1scr ,
per_f1scr ,logR_f1scr ,
SVC_f1scr ,KNC_f1scr ,DTC_f1scr] )

print( ' Best Classifier by F1_score => ',classifiers[max_f1scr])
