import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth,apriori,association_rules

#Input Dateset
org_df = pd.read_csv("amr_horse_ds.csv")

# Binning of Age
org_df['Age'] = pd.cut(org_df['Age'], bins=3, labels=['Young', 'Middle-aged', 'Old'])

#Tabular to Transactional Data
org_df= pd.get_dummies(org_df)

#Extract Association Rules
frequent_patterns_df = fpgrowth(org_df, min_support=0.1,use_colnames=True)
rules_df = association_rules(frequent_patterns_df, metric = "confidence", min_threshold = 0.9)
high_lift_rules_df = rules_df[rules_df['lift'] > 1.5]

#Save Association Rules
high_lift_rules_df.to_csv('arules.csv')

#Visualize Association Rules
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(rules_df['support'], rules_df['confidence'], rules_df['lift'], marker="*")
ax.set_xlabel('support')
ax.set_ylabel('confidence')
ax.set_zlabel('lift')
plt.show()
