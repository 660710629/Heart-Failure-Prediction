import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('heart.csv')
data = pd.get_dummies(data, drop_first=True)
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

anova = SelectKBest(score_func=f_classif, k=15)
X_train_sel= anova.fit_transform(X_train, y_train)
X_test_sel = anova.transform(X_test)

selected_features = X.columns[anova.get_support()]
print("Selected Features:\n", selected_features)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train_sel, y_train)

y_pre = clf.predict(X_test_sel)
cm = confusion_matrix(y_test, y_pre)
print('\nConfusion Matrix:\n', cm)
print('\nClassification Report:\n', classification_report(y_test, y_pre))
