import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/mid/score.csv")

Y = data["label"]
raw_x = data.drop(["sid", "label"], axis=1)


le = LabelEncoder()
raw_x_label = raw_x.apply(le.fit_transform)

enc = OneHotEncoder()
enc.fit(raw_x_label)
X = enc.transform(raw_x_label).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)


clf = RandomForestClassifier(n_estimators=1000, max_features=80)
clf.fit(X_train, y_train)

phat = clf.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, phat[:, 1])
print(metrics.auc(fpr, tpr))
