from _1_data_processing import X_test, Y_test, X_train, Y_train
from _2_model_creating import Y_pred, Y_pred_regr, model_class


#dokładność modelu
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, Y_pred)
print("Dokładność modelu:", accuracy)


#macierz pomyłek
from sklearn.metrics import confusion_matrix

matrix_conf = confusion_matrix(Y_test, Y_pred)
print("Macierz pomyłek:\n", matrix_conf)


#raport klasyfikacji
from sklearn.metrics import classification_report

report_class = classification_report(Y_test, Y_pred)
print("Raport klasyfikacji:\n", report_class)

#ocena jakości regresji
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Błąd średniokwadratowy:", mean_squared_error(Y_test, Y_pred))
print("Średni błąd absolutny:", mean_absolute_error(Y_test, Y_pred))
print("R² (współczynnik determinacji):", r2_score(Y_test, Y_pred))

#współczynnik determinacji
print("R² (Pipeline):", r2_score(Y_test, Y_pred_regr))

#wkaźniki makro - precyzja, czułość
from sklearn.metrics import precision_score, recall_score, f1_score
 ## precyzja i czułość
precision = precision_score(Y_test, Y_pred, average='macro') # TP / (TP + FP)
recall = recall_score(Y_test, Y_pred, average='macro') # TP / (TP + FN)
 ## f1 - średnia harmoniczna dla precyzji i czułości
f1 = f1_score(Y_test, Y_pred, average='macro') # 2 * (precision * recall) / (precision+recall)  

print("Precyzja (makro):", precision)
print("Czułość (makro):", recall)
print("F1-score (makro):", f1)

#krzywa ROC i AUC 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

Y_sample = model_class.predict_proba(X_test)[:, 1]

 ## fpr - False Positive Rate
  ###  fpr = false positives / ( false positives + true negatives)
 ## tpr - True Positive Rate/Sensitivity 
  ###  tpr = true positives = true positives / ( true positives + false negatives )
fpr, tpr, thresholds = roc_curve(Y_test, Y_sample)
aucValue = auc(fpr, tpr)



 ## wizualizacja wykresem
plt.plot(fpr, tpr, label=f'AUC = {aucValue:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Krzywa ROC")
plt.legend()
plt.grid()
plt.show()
