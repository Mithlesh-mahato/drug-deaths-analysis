import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

df = pd.read_csv('C:/Users/anubh/Downloads/Accidental_Drug_Related_Deaths_2012-2024.csv')
df.head()

df.info()
df.shape
df.isnull().sum().sort_values(ascending=False)

df.columns = df.columns.str.strip().str.replace(" ", "_")

mean_data=df['Age'] = df['Age'].fillna(df['Age'].mean())
print(mean_data)

mode_data=df['Sex'] = df['Sex'].fillna(df['Sex'].mode()[0])
print(mode_data)

threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

df['Sex'] = df['Sex'].str.upper().str.strip()
df['Race'] = df['Race'].fillna('Unknown').str.upper().str.strip()

drug_cols = ['Fentanyl', 'Heroin', 'Cocaine']

for col in drug_cols:
    if col in df.columns:
        df[col] = df[col].map({'Y':1, 'N':0})

df['Fentanyl'] = df['Fentanyl'].fillna(0).astype(int)


df.drop(['Death_State','Death_County','Death_City','Injury_County'], axis=1, inplace=True, errors='ignore')

df['Any_Opioid'] = df['Any_Opioid'].fillna('Unknown')
df['Residence_State'] = df['Residence_State'].fillna('Unknown')
df['Injury_State'] = df['Injury_State'].fillna('Unknown')

cols = ['Residence_County','Description_of_Injury','Residence_City']
for col in cols:
    df[col] = df[col].fillna('Unknown')
df['Manner_of_Death'] = df['Manner_of_Death'].fillna('Unknown')

df.drop(['Location'], axis=1, inplace=True)

df['Injury_City'] = df['Injury_City'].fillna('Unknown')
df['Injury_Place'] = df['Injury_Place'].fillna('Unknown')

df['ResidenceCityGeo'] = df['ResidenceCityGeo'].fillna('Unknown')
df['InjuryCityGeo'] = df['InjuryCityGeo'].fillna('Unknown')
df['DeathCityGeo'] = df['DeathCityGeo'].fillna('Unknown')

bins = [0,18,30,45,60,100]
labels = ['Teen','Young','Adult','Mid','Senior']

df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

df = df[df['Sex'] != 'UNKNOWN']
df['Age_Group'] = pd.Categorical(df['Age_Group'], 
                                categories=['Teen','Young','Adult','Mid','Senior'], 
                                ordered=True)
df['Manner_of_Death'] = df['Manner_of_Death'].str.upper().str.strip()
df['Any_Opioid'] = df['Any_Opioid'].replace({'Y':'Yes','N':'No'})

df['Fentanyl_Label'] = df['Fentanyl'].replace({1:'Yes', 0:'No'})

df[['Fentanyl', 'Fentanyl_Label']].head()

df['Manner_of_Death'] = df['Manner_of_Death'].replace({
    'ACCIDENT': 'ACCIDENTAL'
})


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 15))


plt.subplot(3, 3, 1)
df.groupby('Year').size().plot(marker='o')
plt.title("Deaths per Year")


plt.subplot(3, 3, 2)
sns.countplot(x='Sex', data=df,color="Skyblue")
plt.title("Gender Distribution")


plt.subplot(3, 3, 3)
df['Age'].plot(kind='hist', bins=20)
plt.title("Age Distribution")

plt.subplot(3, 3, 4)
sns.boxplot(x='Sex', y='Age', data=df)
plt.title("Age vs Gender")

# 5. Countplot (Fentanyl)
plt.subplot(3, 3, 5)
sns.countplot(x='Fentanyl_Label', data=df)
plt.title("Fentanyl Involvement")

# 6. Horizontal Bar
plt.subplot(3, 3, 6)
df[df['Residence_State']!='Unknown']['Residence_State'] \
.value_counts().head(10).sort_values().plot(kind='barh')
plt.title("Top States")

# 7. Countplot (Manner of Death)
plt.subplot(3, 3, 7)
sns.countplot(x='Manner_of_Death', data=df)
plt.title("Manner of Death")
plt.xticks(rotation=45)

# 8. Scatter Plot (NEW 🔥)
plt.subplot(3, 3, 8)
sns.scatterplot(x='Age', y='Year', hue='Fentanyl_Label', data=df, alpha=0.6)
plt.title("Age vs Year")

# 9. Heatmap
plt.subplot(3, 3, 9)
sns.heatmap(df.corr(numeric_only=True), annot=True,cmap="coolwarm")
plt.title("Correlation")

plt.tight_layout()
plt.show()

X = pd.get_dummies(
    df[['Age','Year','Month','Sex','Race','Manner_of_Death']], 
    drop_first=True
)

y = df['Fentanyl']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.9, random_state=42
)

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)   # ✅ correct

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix (Gradient Boosting)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred_rf))

sns.countplot(x='Fentanyl_Label', hue='Any_Opioid', data=df)
plt.title("Fentanyl vs Opioid Relationship")

df.groupby(['Year','Fentanyl_Label']).size().unstack().plot()
plt.title("Fentanyl Trend Over Years")

models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
accuracy = [0.79, 0.73, 0.76]

plt.bar(models, accuracy)
plt.title("Model Comparison")
plt.ylabel("Accuracy")



























































































































































