from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import  pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

data= pd.read_csv('heart.csv')

data= pd.get_dummies(data, drop_first=True)

X = data.drop("HeartDisease", axis=1)   # Features
y = data["HeartDisease"]                # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1,
    max_depth=4, random_state=42).fit(X_train, y_train)
clf.score(X_test, y_test)
y_pre = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pre)
print('Confusion Matrix:\n', cm)
print('\nClassification Report:\n', classification_report(y_test, y_pre))
