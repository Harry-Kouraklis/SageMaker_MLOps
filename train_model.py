import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

if __name__ == "__main__":
#     X_train_features_output_path = os.path.join("/opt/ml/processing/", "X_train.csv")
#     X_test_features_output_path = os.path.join("/opt/ml/processing/", "X_test.csv")
#     y_train_features_output_path = os.path.join("/opt/ml/processing/", "y_train.csv")
#     y_test_features_output_path = os.path.join("/opt/ml/processing/", "y_test.csv")
    
    parser = argparse.ArgumentParser()


    # Add the default SageMaker arguments
    parser.add_argument('X_train')
    parser.add_argument('y_train')
    
    args = parser.parse_args()

    train_data_path = args.train
    
    print("Reading input data")
    X_train = pd.read_csv(X_train_features_output_path, header=None)
    y_train = pd.read_csv(y_train_features_output_path, header=None)
    

    model = LogisticRegression(class_weight="balanced", solver="lbfgs")
    
    print("Training LR model")
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print(score)
    
    model_output_directory = os.path.join("/opt/ml/model", "model.joblib")
    
    print(f"Saving model to {model_output_directory}")
    joblib.dump(model, model_output_directory)
    
    # can probably find other way to do logging 
    
    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print(f"Saving classification report to {evaluation_output_path}")

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(
            {
            "score": score
        }
        )
               )