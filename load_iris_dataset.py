from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

data = load_iris()
X = data["data"]
Y = data["target"]

X = normalize(X, norm='max')
X, Y = shuffle(X, Y, random_state=0)
# X_train, X_test, y_train, y_train = train_test_split(X,Y, test_size=0.2, random_state=0)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=0)


