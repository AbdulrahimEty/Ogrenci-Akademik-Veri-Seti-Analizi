import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fontTools.misc import transform

df = pd.read_csv('data/student_performance.csv')

pd.set_option('display.max_columns', None)

#print(df.head())

# Student_id kolonunu gereksiz olduğu için passed kolonunu da regresyon modeli yapacağım için siliyorum

df = df.drop(['passed', 'student_id'], axis=1)

#print(df.head())
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())

#print(df['parent_education'].unique())



# Parent educationdaki none değerleri en çok tekrar eden değerle değiştiriyorum
#print(df['parent_education'].mode())
df['parent_education'] = df['parent_education'].fillna(df['parent_education'].mode()[0])

#print(df.head())
#print(df.isnull().sum())


map_cols = ['extracurricular', 'internet_access']

for i in map_cols:
    df[i] = df[i].map({"Yes": 1, "No": 0})


df['gender'] = df['gender'].map({"Female": 0, "Male": 1})

#print(df.head())

#plt.figure(figsize=(12,8))
#sns.histplot(df['final_score'], bins=50)
#plt.show()



#print(df.describe())

#df.hist(bins=50)
#plt.tight_layout()
#plt.show()

#plt.figure(figsize=(15,8))
#sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
#plt.tight_layout()
#plt.show()

#sns.scatterplot(data=df, x="final_score", y="study_hours_per_week", hue="gender")
#plt.xlabel("Final Score")
#plt.ylabel("Study Hours per Week")
#plt.legend(loc="upper left")
#plt.show()

X = df.drop('final_score', axis=1)
y = df['final_score']

new_cols = ['parent_education', 'gender', 'age', 'study_hours_per_week',
            'attendance_rate', 'internet_access', 'extracurricular', 'previous_score']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.compose import ColumnTransformer

#print(df['parent_education'].unique())

ct = ColumnTransformer(transformers=[

    ('ord', OrdinalEncoder(categories=[['High School', 'Bachelor', 'Master', 'PhD']]), ['parent_education']),
], remainder='passthrough')

X_train_transform = ct.fit_transform(X_train)
X_test_transform = ct.transform(X_test)

X_train_transform = pd.DataFrame(X_train_transform, columns=new_cols)

#print(X_train_transform.head())

#print(X_train_transform['parent_education'].unique())

print(X_train_transform.shape)
print(X_train_transform[:3])

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_transform)
X_test_scaled = scaler.transform(X_test_transform)

from lazypredict.Supervised import LazyRegressor

#reg = LazyRegressor(verbose=0, ignore_warnings=True)
#models, predictions = reg.fit(X_train_scaled, X_test_scaled, y_train, y_test)
#print(models)

from sklearn.linear_model import Ridge, Lasso

# Ridge tuning
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(ridge, params, cv=5, scoring='r2')
grid.fit(X_train_scaled, y_train)

print(grid.best_params_)
print(grid.best_score_)


best_ridge = grid.best_estimator_
y_pred = best_ridge.predict(X_test_scaled)

from sklearn.metrics import r2_score, mean_squared_error
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


coef_df = pd.DataFrame({
    'feature': X_train_transform.columns,
    'coefficient': best_ridge.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(coef_df)