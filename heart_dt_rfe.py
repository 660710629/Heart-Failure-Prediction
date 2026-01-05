from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import pandas as pd

data = pd.read_csv('heart.csv')
data = pd.get_dummies(data, drop_first=True)

X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimator = DecisionTreeClassifier(random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=10)  # เลือก 10 features
rfe.fit(X_train, y_train)

selected_features = X.columns[rfe.support_]
print("\nSelected Features:")
print(selected_features)

X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train_selected, y_train)

y_pred = clf.predict(X_test_selected)

cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:\n', cm)
print('\nClassification Report:\n', classification_report(y_test, y_pred))
