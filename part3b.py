import pandas as pd
import part3 as p3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn import decomposition


def normalize(df, features):
    result = df.copy()
    for feature_name in features:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


data = p3.data_df()
clean_data = p3.transform_data(data)
'''
list1 = ['avg_dist', 'avg_surge', 'trips_in_first_30_days',
         'avg_rating_by_driver', 'avg_rating_of_driver', 'weekday_pct']
clean_data = normalize(clean_data, [list1])
clean_data['avg1'] = clean_data[['avg_dist', 'avg_surge', 'trips_in_first_30_days']].mean(axis=1)
clean_data['avg2'] = clean_data[['avg_rating_by_driver', 'avg_rating_by_driver']].mean(axis=1)
clean_data['fea1'] = clean_data['city'] * clean_data['phone']
clean_data.drop(list1, inplace=True, axis=1)
clean_data.drop(['city', 'phone'], inplace=True, axis=1)
# clean_data['avg'] = clean_data[['avg1', 'avg2']].mean(axis=1)
# clean_data.drop(['avg2'], inplace=True, axis=1)
# print(clean_data.head())
'''
pca = decomposition.PCA(n_components=3, svd_solver='full')
pca.fit(clean_data.drop(['active'], axis=1))
x = pca.transform(clean_data.drop(['active'], axis=1))

y = clean_data['active']

x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.999, stratify=y)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

# score
print('Feature importance')
print(clf.feature_importances_)
print('F1 score {}'.format(f1_score(y_test, y_pred)))
print('Precision {}'.format(precision_score(y_test, y_pred)))
print('Recall {}'.format(recall_score(y_test, y_pred)))
print('Test Accuracy {}'.format(accuracy_score(y_test, y_pred)))
print(confusion_matrix(y_test, y_pred))
