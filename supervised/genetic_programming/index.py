from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np 

#Read the CSV
data =  pd.read_csv('GlobalLandTemperaturesByCountry.csv',delimiter=",",skip_blank_lines=True,verbose=True).dropna(how='any')

#Encoding Countries into Nodes
label_encoder = LabelEncoder().fit(data['Country'])
data['Country'] = label_encoder.transform(data['Country'])
#country = label_encoder.tranform(data['Country'])

data = data.astype(np.float64)
X_train , X_test , Y_train , Y_test = train_test_split(data.drop('Country',axis=1),
data['Country'], train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5,verbosity=2)

print(len(X_train),len(Y_train))
tpot.fit(X_train,Y_train)

tpot.score(X_test,Y_test)

tpot.export('optimised.py')
