import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import missingno as msno  # kayıp veri analizi

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn import tree

df = pd.read_csv("water_potability.csv")

describe = df.describe()
print(describe)

d = pd.DataFrame(df["Potability"].value_counts())
fig = px.pie(
    d,
    values="count",
    names=["Not Potable", "Potable"],
    hole=0.35,
    opacity=0.8,
    labels={"label": "Potability", "Potability": "Number of Samples"},
)
fig.update_layout(title=dict(text="Pie Chart of Potability Feature"))
fig.update_traces(textposition="outside", textinfo="percent+label")
fig.show()
fig.write_html("potability_pie_chart.html")

# dağılım analizi
non_potable = df.query("Potability==0")
potable = df.query("Potability==1")

plt.figure(figsize=(14, 10))
for ax, col in enumerate(df.columns[:9]):
    plt.subplot(3, 3, ax + 1)
    plt.title(col)
    sns.kdeplot(x=non_potable[col], label="Non Potable")
    sns.kdeplot(x=potable[col], label="Potable")
    plt.legend()
plt.tight_layout()
plt.show()

msno.matrix(df)
plt.show()

# eksik verileri ortalama ile doldurma
df["ph"] = df["ph"].fillna(value=df["ph"].mean())
df["Sulfate"] = df["Sulfate"].fillna(value=df["Sulfate"].mean())
df["Trihalomethanes"] = df["Trihalomethanes"].fillna(value=df["Trihalomethanes"].mean())

X = df.drop("Potability", axis=1).values
y = df["Potability"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# sütun bazında min-max normalizasyon
x_train_max = np.max(X_train, axis=0)
x_train_min = np.min(X_train, axis=0)
X_train = (X_train - x_train_min) / (x_train_max - x_train_min)
X_test = (X_test - x_train_min) / (x_train_max - x_train_min)

models = [
    ("DTC", DecisionTreeClassifier(max_depth=10, random_state=42)),
    ("RF", RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)),
]

finalResult = []
cmlist = []
for name, model in models:
    model.fit(X_train, y_train)
    model_result = model.predict(X_test)

    acc = accuracy_score(y_test, model_result)
    prec = precision_score(y_test, model_result)
    rec = recall_score(y_test, model_result)
    f1 = f1_score(y_test, model_result)

    finalResult.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
    })

    cm = confusion_matrix(y_test, model_result)
    cmlist.append((name, cm))

results_df = pd.DataFrame(finalResult)
print(results_df.to_string(index=False))

for name, cm in cmlist:
    plt.figure()
    sns.heatmap(cm, annot=True, linewidths=0.8, fmt=".0f")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

dt_clf = models[0][1]
plt.figure(figsize=(20, 12))
tree.plot_tree(
    dt_clf,
    feature_names=df.columns.tolist()[:-1],
    class_names=["0", "1"],
    filled=True,
    precision=2,
    max_depth=3,
)
plt.show()