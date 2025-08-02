from _1_data_processing import X_test, Y_test, X_train, Y_train
from _2_model_creating import Y_pred, Y_pred_regr, model_class


#przeszukiwanie siatki - search grid

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, Y_train)
best_model = grid_search.best_estimator_


#losowy tuning - randomized search
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

random_search = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=10, cv=5)
random_search.fit(X_train, Y_train)

#ręczne strojenie - manual tuning

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear')
model.fit(X_train, Y_train)



#optymalizacja Bayes'a - Bayes optimalization

from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier

## definiowanie przestrzeni hiperparametrów
parametry = {
    'n_estimators': (50, 300),
    'max_depth': (5, 50),
    'min_samples_split': (2, 20)
}

## konfiguracja optymalizacji
bayes_opt = BayesSearchCV(
    estimator=RandomForestClassifier(),
    search_spaces=parametry,
    n_iter=32,  # liczba iteracji (punktów do przetestowania)
    cv=5,       # walidacja krzyżowa
    random_state=42,
    scoring='accuracy'
)

## trening z optymalizacją 
bayes_opt.fit(X_train, Y_train)

## najlepsze parametry
print("Best params:", bayes_opt.best_params_)
print("score:", bayes_opt.best_score_)
