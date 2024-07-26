import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow

mlflow.set_tracking_uri('https://dagshub.com/HimanshuBhardwaj174/dagshub-mlflow.mlflow')
import dagshub
dagshub.init(repo_owner='HimanshuBhardwaj174', repo_name='dagshub-mlflow', mlflow=True)



iris = load_iris()
X = iris.data
y = iris.target
n_estimators=100
random_state=42
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
mlflow.set_experiment('iris-dt')
with mlflow.start_run():
    # Initialize the RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    cm = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('conf.png')
    mlflow.log_params({
        'n_estimator ':n_estimators,
        'random_state': random_state
    })

    mlflow.log_metric('accuracy',accuracy)

    #mlflow.sklearn.log_model(clf,'rf')
    mlflow.log_artifact('conf.png')
    mlflow.log_artifact(__file__)

