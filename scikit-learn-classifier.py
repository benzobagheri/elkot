# import library from Scikit-Learn ---------------------------------------------
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# algorithm 1 ------------------------------------------------------------------
print(" Naive Bayes ... ")

start = timeit.default_timer()
from sklearn import naive_bayes
classifier = naive_bayes.GaussianNB()
nb_model = classifier.fit(X, Y)
prediction = nb_model.predict(X_test)
end = timeit.default_timer()

print(" accuracy = ", accuracy_score(Y_test, prediction), " time = ", end - start)
print(confusion_matrix(Y_test, prediction))
print("\n")

# algorithm 2 ------------------------------------------------------------------
print(" Random Forest ... ")

start = timeit.default_timer()
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
rf_model = classifier.fit(X, Y)
prediction = rf_model.predict(X_test)
end = timeit.default_timer()

print(" accuracy = ", accuracy_score(Y_test, prediction), " time = ", end - start)
print(confusion_matrix(Y_test, prediction))
print("\n")

# algorithm 3 ------------------------------------------------------------------
print(" Gradient Boosting ... ")

start = timeit.default_timer()
from sklearn.ensemble import GradientBoostingClassifier as gbc
classifier = gbc()
gbc_model = classifier.fit(X, Y)
prediction = gbc_model.predict(X_test)
end = timeit.default_timer()

print(" accuracy = ", accuracy_score(Y_test, prediction), " time = ", end - start)
print(confusion_matrix(Y_test, prediction))
print("\n")

# algorithm 4 ------------------------------------------------------------------
print(" SVM ... ")

start = timeit.default_timer()
from sklearn import svm
classifier = svm.SVC()
svc_model = classifier.fit(X, Y)
prediction = svc_model.predict(X_test)
end = timeit.default_timer()

print(" accuracy = ", accuracy_score(Y_test, prediction), " time = ", end - start)
print(confusion_matrix(Y_test, prediction))
print("\n")
