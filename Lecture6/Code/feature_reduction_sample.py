from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# Load data
org_df = pd.read_csv("diabetes_Truncated.csv", index_col=0)

#Define features and label
label_df =  org_df.loc[:,org_df.columns == 'Outcome']
feat_df =  org_df.loc[:,org_df.columns != 'Outcome']

# Normalize all features
norm_feat_df = (feat_df - feat_df.mean()) / feat_df.std()


# Mutual Info
mutual_info_gl_age = mutual_info_regression(feat_df[['Glucose']], feat_df['Age'])
mutual_info_bp_bmi = mutual_info_regression(feat_df[['BloodPressure']], feat_df['BMI'])
print('Mutual info between Gl and Age:',mutual_info_gl_age)
print('Mutual info between BP and BMI:',mutual_info_bp_bmi)

# Correlation Matrix
corr_matrix = org_df.corr()
print(corr_matrix)

# Variance threshold
varModel=VarianceThreshold(threshold=0.7)
varModel.fit(norm_feat_df)
print(varModel.variances_)
selected_features = varModel.get_feature_names_out()
print(selected_features)


# Wrapper methods

#KNN Model as classifier
knn = KNeighborsClassifier(n_neighbors=9)

#Forward
sfs = SFS(knn, k_features='best', forward=True,scoring='accuracy')
sfs.fit(norm_feat_df, label_df)
sfs_metric_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
print(sfs_metric_df)
selected_features_df = feat_df[list(sfs_metric_df['feature_names'][3])]
print(selected_features_df)


#Backward
sbs = SFS(knn, k_features='best', forward=False, scoring='accuracy')
sbs.fit(norm_feat_df, label_df)
sbs_metric_df = pd.DataFrame.from_dict(sbs.get_metric_dict()).T
print(sbs_metric_df)


# PCA

# Create and train PCA model
model = PCA()
model.fit(norm_feat_df)

# Show covariance matrix
print(model.get_covariance())

# Show eigenvalues and eigenvectors
pc_feature_relationship = pd.DataFrame(model.components_,  columns=norm_feat_df.columns)
print(pc_feature_relationship)


# Plot the proportion of explained variance by Cumulative sum of principal components
x_axis = range(model.n_components_)
plt.plot(x_axis, np.cumsum(model.explained_variance_ratio_), marker="o")
plt.xlabel('Principal component')
plt.ylabel('Cumulative sum of explained variance')
plt.xticks(x_axis)
plt.show()

#
# Prediction by original variables
# Predict using knn
knn.fit(norm_feat_df, label_df)
y_pred = knn.predict(norm_feat_df)

#Display the accuracy
accuracy = accuracy_score(label_df, y_pred)
print('Accuracy by Original Features:', accuracy )


# Prediction by variables made using PCA
# Create and train PCA model
pca = PCA(n_components=3)
pca.fit(norm_feat_df)

# Transform existing features to new features
transformed_features = pca.transform(norm_feat_df)

# Predict using new features
knn.fit(transformed_features, label_df)
y_pred = knn.predict(transformed_features)

#Display the accuracy
accuracy = accuracy_score(label_df, y_pred)
print('Accuracy by PCA:', accuracy )


# Prediction by variables made using LDA
# Create and train LDA model
lda = LDA(n_components=1)
lda.fit(norm_feat_df, label_df)

# Transform existing features to new features
transformed_features = lda.transform(norm_feat_df)

# Predict using new features
knn.fit(transformed_features, label_df)
y_pred = knn.predict(transformed_features)

#Display the accuracy
accuracy = accuracy_score(label_df, y_pred)
print('Accuracy by LDA:', accuracy )
