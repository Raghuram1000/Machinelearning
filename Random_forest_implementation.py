import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def train_evaluate_random_forest(csv_file):
    # Load the data from CSV
    data = pd.read_csv(csv_file)

    # Prepare the data
    X = data.drop('class', axis=1)
    y = data['class']

    # Perform one-hot encoding on categorical columns
    categorical_columns = X.select_dtypes(include='object').columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]))
    X_encoded.columns = encoder.get_feature_names_out(categorical_columns)
    X = pd.concat([X.drop(categorical_columns, axis=1), X_encoded], axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Building the Classifier
    random_forest = RandomForestClassifier(random_state=50)

    # Training
    random_forest.fit(X_train, y_train)

    # Predictions
    y_randomforest_predictions = random_forest.predict(X_test)

    # Evaluation Metrics of the model Accuracy and F1 score
    accuracy = accuracy_score(y_test, y_randomforest_predictions)
    f1 = f1_score(y_test, y_randomforest_predictions, pos_label=' >50K.')
    print('Accuracy score:', round(accuracy * 100, 2))
    print('F1 score:', round(f1 * 100, 2))

# Usage
train_evaluate_random_forest('discritiexefadfas.csv')
train_evaluate_random_forest('discritiexefadfas(1).csv')
