from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd



data = pd.read_csv('modified_file2.csv')

continuous_attributes = ['Age', 'fnlwgt', 'Education-Num','Capital-Gain','Capital-Loss','Hours-per-week']



def discretize_data(inp_df, continuous_features):
    discretized_df = inp_df.copy()
    # use kbnins discretizer to discretize the continuous features
    discritizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')

    for feature in continuous_features:
        discretized_df[feature] = discritizer.fit_transform(inp_df[feature].values.reshape(-1, 1))

    # for each feature, replace 0, 1, 2, 3, or 4 with your own labels
    for feature in continuous_features:
        discretized_df[feature] = discretized_df[feature].replace([0, 1, 2, 3, 4],
                                                                  ['very-low', 'low', 'medium', 'high', 'very-high'])

    return discretized_df

res = discretize_data(data, continuous_attributes)

res.to_csv('discritiexefadfas.csv', index=False)