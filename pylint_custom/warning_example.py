"""
file description placeholder
"""
path_parent = 'parent'
path_full = f"{path_parent}/child"

data_path_hardcoded = "/data/dataset.csv"

from sklearn.model_selection import train_test_split
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 1, 0, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y)
