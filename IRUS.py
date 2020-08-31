from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
print("-------------iris.data-------------")
print(iris.data[0:5])


print()

print("-------------iris.target-------------")
print(iris.target)




X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

tpot = TPOTClassifier(verbosity=2, max_time_mins=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

print("------------------------TPOT Finished--------------------------")
