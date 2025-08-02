from _1_data_processing import X_test, Y_test, X_train, Y_train

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# inicjalizacja modelu klasyfikacyjnego
model_class = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

 ## trenowanie modelu na danych treningowych
model_class.fit(X_train, Y_train)

 ## predykcja na zbiorze testowym
Y_pred = model_class.predict(X_test)


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

pipeline_model_regr = Pipeline([
    ('imputacja', SimpleImputer(strategy='median')),
    ('skalowanie', StandardScaler()),
    ('selekcja', SelectKBest(score_func=f_classif, k=10)),
    ('klasyfikator', RandomForestClassifier(random_state=42))
])

pipeline_model_regr.fit(X_train, Y_train)
Y_pred_regr = pipeline_model_regr.predict(X_test)

print("Dokładność pipelinu:", accuracy_score(Y_test, Y_pred_regr))

#inicjalizacja modelu regresyjnego

from sklearn.linear_model import LinearRegression

model_regr = LinearRegression()

 ## trenowanie modelu
model_regr.fit(X_train, Y_train)

 ## predykcja wartości na zbiorze testowym
Y_pred = model_regr.predict(X_test)


#pipeline dla regresji z preprocessingiem

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipeline_model_regr = Pipeline([
    ('imputacja', SimpleImputer(strategy='median')),
    ('skalowanie', StandardScaler()),
    ('regresja', LinearRegression())
])

pipeline_model_regr.fit(X_train, Y_train)
Y_pred_regr = pipeline_model_regr.predict(X_test)




