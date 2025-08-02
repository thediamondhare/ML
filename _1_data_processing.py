#importy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection, impute, preprocessing 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

#przyjmijmy, że df to dataframe z danymi
df = None

## wymiary df (w, k)
df.shape

## profil danych
df.describe()

## ilość, średnia. odch. stand., min, Q1, Q2, Q3, max 
result = df.describe().iloc[:,:2]


#wyszukiwanie brakujących danych
df.isnull().any(axis=1)
##podusowanie ile NaN w każdej kolumnie
df.isnull().sum() 
##odsetek brakujących danych
df.isnull().mean() * 100

#zliczenie unikalnych wartości
df.cecha.value_counts(dropna=False)

#usunięcie kolumn, gdy są mocno skorelowane
df.drop(columns=['skorelowana1', 'skorelowana2'])

Y = df.cechaAnalizowana
X = df.drop(columns='cechaAnalizowana')


#wyodrębnienie 30% danych do testu/treningu modelu
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y , test_size=0.3, random_state=42
    )


#imputacja danych

kolumny = [
    'cecha1',
    'cecha2',
    'cecha3',
    'cecha4',
]

imputer = impute.IterativeImputer()

imputed = imputer.fit_transform(X_train[kolumny])
X_train.loc[:,  kolumny] = imputed

imputed = imputer.transform(X_test[kolumny])
X_test.loc[:,  kolumny] = imputed


#imputacja danych przy wykorzystaniu mediany
med =X_train.median()
X_train = X_train.fillna(med)
X_test = X_train.fillna(med)


#normalizacja
sca = preprocessing.StandardScaler()
X_train = sca.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=kolumny)

X_test = sca.fit_transform(X_test)
X_test = pd.DataFrame(X_test, columns=kolumny)


#wybór cech / redukcja
selection = SelectKBest(score_func=f_classif, k=10)
X_train_selection = selection.fit_transform(X_train, Y_train)
X_test_selection = selection.transform(X_test)

#budowa Pipeline
model = Pipeline([
    ('imputacja', impute.SimpleImputer(strategy='median')),
    ('normalizacja', preprocessing.StandardScaler()),
    ('klasyfikator', RandomForestClassifier())
])

model.fit(X_train, Y_train)


#detekcja wartyości odstających (outliers)
Q1 = df['cechaLiczbowa'].quantile(0.25)
Q3 = df['cechaLiczbowa'].quantile(0.75)
IQR = Q3 - Q1

outlier_mask = ( df['cechaLiczbowa'] < Q1 - 1.5 * IQR ) | (df['cechaLiczbowa'] > Q3 + 1.5 * IQR)
df_outliers = df[outlier_mask]

#kodowanie zmiennych kategorycznych
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['cechaKategoryczna']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['cechaKategoryczna']))
df = pd.concat([df.drop(columns='cechaKategoryczna'), encoded_df], axis=1)

