import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
data = pd.read_csv('crop_dataset.csv')
data.head()
data.shape
X = data.iloc[:,:-1]
X.head()
y = data.iloc[:,-1]

y.head()
data['label'].value_counts()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=1)
sns.countplot(x='label',data=data)
plt.show()
X_train.shape
X_train.head()
y_test.shape
y_test.head()


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train,y_train)
filename = 'model1.sav'
pickle.dump(model, open(filename, 'wb'))
y_pred = model.predict(X_test)
from sklearn import metrics
acc=(metrics.accuracy_score(y_pred,y_test))
print("Accuracy is:",acc)
print("Confusion Matrix is: ",metrics.confusion_matrix(y_pred,y_test))

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

