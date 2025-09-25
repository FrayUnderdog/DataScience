import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. load data
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 2.  72% train, 20% test, 8% validation 划
# divide into train+valid & test
X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# divide validation from train+valid 
X_train, X_val, y_train, y_val = train_test_split(
    X_train_valid, y_train_valid, test_size=0.10/0.80, random_state=42, stratify=y_train_valid
)

print("Train size:", len(X_train), "Validation size:", len(X_val), "Test size:", len(X_test))

# 3. grid-search for hyperparameters
param_grid = {
    "max_depth": [3, 5],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [3, 5],
    "min_impurity_decrease": [0.01, 0.001],
    "ccp_alpha": [0.001, 0.0001]
}

best_model = None
best_acc = 0
best_params = None

for depth in param_grid["max_depth"]:
    for split in param_grid["min_samples_split"]:
        for leaf in param_grid["min_samples_leaf"]:
            for impurity in param_grid["min_impurity_decrease"]:
                for alpha in param_grid["ccp_alpha"]:
                    clf = DecisionTreeClassifier(
                        criterion="gini",
                        max_depth=depth,
                        min_samples_split=split,
                        min_samples_leaf=leaf,
                        min_impurity_decrease=impurity,
                        ccp_alpha=alpha,
                        random_state=42
                    )
                    clf.fit(X_train, y_train)
                    val_pred = clf.predict(X_val)
                    acc = accuracy_score(y_val, val_pred)
                    if acc > best_acc:
                        best_acc = acc
                        best_model = clf
                        best_params = (depth, split, leaf, impurity, alpha)

print("Best Validation Accuracy:", best_acc)
print("Best Params:", best_params)

# 4. best-params on test set
test_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print("Test Accuracy:", test_acc)

# 5. extract 3 rules
rules = export_text(best_model, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n")
print(rules)

# ================= Decision Tree Extracted Rules =================
# Rule 1:
# if BMI <= 29.45 and Age <= 49.5 and Glucose <= 130.5:
#     Outcome = 0   # no diabetes
#
# Rule 2:
# if BMI > 29.45 and Age > 27.5 and Glucose <= 147.0:
#     Outcome = 1   # diabetes
#
# Rule 3:
# if BMI > 29.45 and Age <= 27.5 and Glucose <= 97.5:
#     Outcome = 1   # diabetes
# ================================================================


rule_lines = rules.split("\n")
for i, line in enumerate(rule_lines[:15]):  # print first 15 lines to check the structure
    print(line)
