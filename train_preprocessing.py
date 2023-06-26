from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = load_wine()

data = wine['data']
target = wine['target']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)


X_train_features_output_path = os.path.join("/opt/ml/processing/", "X_train.csv")
X_test_features_output_path = os.path.join("/opt/ml/processing/", "X_test.csv")
y_train_features_output_path = os.path.join("/opt/ml/processing/", "y_train.csv")
y_test_features_output_path = os.path.join("/opt/ml/processing/", "y_test.csv")

pd.DataFrame(X_train).to_csv(X_train_features_output_path, header=False, index=False)
pd.DataFrame(X_test).to_csv(X_test_features_output_path, header=False, index=False)
pd.DataFrame(y_train).to_csv(y_train_features_output_path, header=False, index=False)
pd.DataFrame(y_test).to_csv(y_test_features_output_path, header=False, index=False)

# good idea to save as joblib the processor itself as transformer