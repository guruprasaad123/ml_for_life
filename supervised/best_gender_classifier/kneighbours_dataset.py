import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score


# Import the adult.txt file into Python
data = pd.read_csv('adults.txt', sep=',')

# DO NOT WORRY ABOUT THE FOLLOWING 2 LINES OF CODE
# Convert the string labels to numeric labels

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

# Instantiate the KNeighboursClassifier classifier
clf =  KNeighborsClassifier()

# Train the classifier using the train data
clf = clf.fit(X_train, Y_train)

# Validate the classifier
accuracy = clf.score(X_test, Y_test)
print( 'Accuracy: ' + str(accuracy))

# Make a confusion matrix
prediction = clf.predict(X_test)

cm = confusion_matrix(prediction, Y_test)
print( cm)
#print('prediction => ',prediction)
#print('Correct => ',Y_test)
f1score = f1_score(prediction,Y_test)
precisionscore = precision_score(prediction,Y_test)
recallscore = recall_score(prediction,Y_test)

print('Precision Score => ',precisionscore)
print('recall Score => ',recallscore)
print('f1 Score => ',f1score)