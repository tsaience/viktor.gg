import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = np.arange(10).reshape((5, 2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42)

print(X, y)

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

example_predict = lr_model.predict(X_test)

print(X_test)
print(y_test)
print(example_predict)