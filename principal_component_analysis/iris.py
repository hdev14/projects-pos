from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris_dataset = datasets.load_iris()
feature_names = iris_dataset.feature_names

df = pd.DataFrame(iris_dataset.data, columns=feature_names)
df['Target'] = iris_dataset.get('target')

X = df[feature_names].values
y = df['Target'].values

X = StandardScaler().fit_transform(X)

df_standard = pd.DataFrame(data=X, columns=feature_names)

print(df_standard.head())

pca = PCA(n_components=2)

principal_components = pca.fit_transform(X)

df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

target = pd.Series(iris_dataset['target'], name='target')
result_df = pd.concat([df_pca, target], axis=1)
print(result_df)

print('Variance of each component:', pca.explained_variance_ratio_)
print('Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_))*100, 2))