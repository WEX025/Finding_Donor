import pandas as pd
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score
from sklearn.grid_search import GridSearchCV
from time import time

# Load dataset
data = pd.read_csv("census.csv")

# Split dataset
income = map(lambda x: 0 if x=='<=50K' else 1, data.income)

features_raw = data.drop('income', axis=1)
non_numeric = ['workclass','education_level','marital-status','occupation','relationship','race','sex','native-country']
features = pd.get_dummies(data=features_raw, columns=non_numeric)

X_train, X_test, y_train, y_test = train_test_split(features, income, stratify=income, test_size=0.2, random_state = 42)

# Model fitting
model = XGBClassifier()
model.fit(X_train, y_train)

start = time()
model_pred = model.predict(X_test)
end = time()
pred_time = end - start


# Model tuning
start = time()
parameters = {'max_depth':[2,4,6,8,10],'min_child_weight':[0,2,4,6], 'gamma':[0,2,4],'max_delta_step':[0,3,6,9]}
scorer = make_scorer(fbeta_score, beta=0.5)
grid_fit = GridSearchCV(model, parameters, cv=3, scoring=scorer).fit(X_train, y_train)
best_model = grid_fit.best_estimator_
end = time()
tune_time = end - start

# Scoring
results = {}
results['accuracy_unoptimized']=accuracy_score(y_test, model.predict(X_test))
results['f_score_unoptimized']=fbeta_score(y_test, model.predict(X_test),beta=0.5)
results['accuracy_optimized']=accuracy_score(y_test, best_model.predict(X_test))
results['f_score_optimized']=fbeta_score(y_test, best_model.predict(X_test),beta=0.5)
print best_model
print results
print "Predition time is:", pred_time, "sec, and tuning time is:", tune_time, "sec"
