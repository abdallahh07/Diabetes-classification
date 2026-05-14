import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    classification_report
)

# 1. load the file 
f = pd.read_csv("diabetes.csv")

# 2. Review the data
print(f.describe())
print(f.info())
print(f.dtypes)
print(f.isna().sum())
print(f.columns)

# 3. analyze 
def analyze (f) :
  fig,ax = plt.subplots(3,2,figsize=(15,10))
  ax = ax.flatten()
  
  # 3.1 Age distribution
  sns.histplot(data=f,
              x="Age",
              bins=10,
              kde=True,
              ax=ax[0])

  ax[0].set_title("Age Distribution")
  
  # 3.2 the relation between the age and diabetes
  sns.histplot(data=f,
               x="Age",
               y="DiabetesPedigreeFunction", 
               ax=ax[1])
  ax[1].set_title("relation between the age and diabetes")
  
  # 3.3 Diabetes Outcome Count
  sns.countplot(data=f,
                x="Outcome",
                ax=ax[2])
  ax[2].set_title("Diabetes Outcome Count")
  ax[2].set_xticklabels(["No Diabetes","Diabetes"])
  
  #3.4 Glucose vs Outcome
  sns.boxplot(data=f,
              x="Outcome",
              y="Glucose",
              ax=ax[3])
  ax[3].set_title("Glucose vs Outcome")
  ax[3].set_xticklabels(["No Diabetes","Diabetes"])
  
  # 3.5 BMI distribution
  sns.histplot(data=f,
               x="BMI",
               bins=20,
               kde=True,
               ax=ax[4])
  ax[4].set_title("BMI distribution")
  
  # 3.6 Correlation Heat Map
  sns.heatmap(f.corr(),
              annot=True,
              cmap="coolwarm",
              ax=ax[5])
  ax[5].set_title("Correlation heatmap")  
  plt.tight_layout(pad=3.0)
  
  plt.figure(figsize=(14,6))
  stat = f.describe().loc[["mean","std","min","max"]]
  sns.heatmap(
    stat,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    linewidths=0.5)
   
  plt.show()

analyze(f)

# 4 define features and target
x = f.drop("Outcome", axis=1)
y = f["Outcome"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# 5 the models 
models = {
  "RandomForestClassifier" : RandomForestClassifier(),
  "KNeighborsClassifier" : KNeighborsClassifier(),
  "LogisticRegression" : LogisticRegression(),
  "XGBClassifier" : XGBClassifier(),
  "SVC" : SVC(probability= True),
  "LinearSVC" : LinearSVC()
  }
 
# 6. Train & Evaluate
results = []
best_model_name = None
best_model_accuracy = 0
best_model_pred = None
best_model_cm = None
best_model_report = None


for name,model in models.items():
  pipe=Pipeline([
    ("Scaller",StandardScaler(with_mean=False)),
    ("Polynomial Features",PolynomialFeatures(degree=2,include_bias=False)),
    ("model",model)])
  
  pipe.fit(x_train,y_train)
  pred=pipe.predict(x_test)
  acc= accuracy_score(y_test,pred)
  
  
  print(f"\n{name}")
  print("accuracy_score",acc)
  print("precision_score",precision_score(y_test,pred))
  cm = (confusion_matrix(y_test,pred))
  report = classification_report(y_test,pred)
  print("Confusion_Matrix",cm)
  print("Classification_Report",report)
  
    
  results.append({
    "Model" : name,
    "Accuracy" : acc,})
  
  if acc > best_model_accuracy:
    best_model_accuracy = acc 
    best_model_name = name
    best_model_pred = pred
    best_model_cm = cm
    best_model_report = report
    best_model_pipe = pipe
    
# 8. Visualize the models
model_=pd.DataFrame(results)
plt.figure(figsize=(10,6))
sns.barplot(data=model_,
            x = "Model",
            y = "Accuracy")
plt.title("Models Accuracy Comparison")
plt.tight_layout(pad=3.0)
plt.ylabel("Accuracy")
plt.show()

# 9. visualize the best model
plt.figure(figsize=(6,5))
sns.heatmap(best_model_cm,
            annot=True,
            fmt="d",
            cmap="Blues")
plt.title(f"Confusion matrix-{best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 10. Tune the model
grid_={
  "model__C":[0.1,1,10],
  "model__kernel":["linear","rbf"],
  "model__gamma":["scale","auto"]}

tune_=RandomizedSearchCV(best_model_pipe,param_distributions=grid_,verbose=5)
tune_.fit(x_train,y_train)
print("Best Parameters:",tune_.best_params_)
print("Best Cross Validation Score:",
      tune_.best_score_)

grid_2={
  "model__C":[5,10,15,20],
  "model__kernel":["rbf"],
  "model__gamma":["scale","auto"],
}
tune_2=GridSearchCV(best_model_pipe,param_grid=grid_2,verbose=5)
tune_2.fit(x_train,y_train)
print("Best Params",tune_2.best_params_)
print("Best Cross Validation Score",tune_2.best_score_)
