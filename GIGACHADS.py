#!/usr/bin/env python
# coding: utf-8

# # GIGACHADS

# 

# # First we start with importing all necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import sm as sm

# In[2]:


from matplotlib import pyplot as plt
import plotly


# In[3]:


import seaborn as sns
import plotly.express as px



import statsmodels


from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


# In[11]:


from sklearn.model_selection import train_test_split, GridSearchCV


from sklearn.preprocessing import OneHotEncoder


# In[15]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline


# In[16]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


# In[17]:


import warnings


# In[18]:


# from functions_pkg import print_vif, predictions_df
from sklearn.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)


# In[19]:


from sklearn.calibration import calibration_curve


# In[20]:


# functions from package that won't import into notebook
def print_vif(feature_df):
    """
    Utility for checking multicollinearity assumption
    :param feature_df: input features to check using VIF. This is assumed to be a pandas.DataFrame
    :return: nothing is returned the VIFs are printed as a pandas series
    """
    # Silence numpy FutureWarning about .ptp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feature_df = sm.add_constant(feature_df)

    vifs = []
    for i in range(feature_df.shape[1]):
        vif = variance_inflation_factor(feature_df.values, i)
        vifs.append(vif)


# In[21]:


def predictions_df(X_test, y_test, y_preds):
    """
    Function to create a predictions dataframe from X_test, y_test, y_predictions input

    :param X_test:
    :param y_test:
    :param y_preds: X_test predictions; model.predict(X_test)
    :return pred_df, fig: returns predictions data frame and plotly express fig object
    """

    pred_df = X_test.copy()
    pred_df["y_true"] = y_test
    pred_df["y_preds"] = y_preds
    pred_df["residuals"] = pred_df.y_preds - pred_df.y_true
    pred_df["abs_residuals"] = pred_df.residuals.abs()
    pred_df = pred_df.sort_values("abs_residuals", ascending=False)

    fig = px.scatter(data_frame=pred_df, x="y_true", y="y_preds")
    fig.add_shape(
        type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max()
    )

    return pred_df, fig


# # Load the Data

# In[22]:


df=pd.read_csv('sample_data.csv')
df


# In[23]:


df.describe()


# In[24]:


df.info()


# # Data Cleaning

# In[25]:


# column renaming
mapping = {
    "ap_hi": "bp_hi",
    "ap_lo": "bp_lo",
    "gluc": "glucose",
    "alco": "alcohol",
    "cardio": "disease",
}

df = df.rename(columns=mapping)



# # Checking missing values

# In[27]:


# no null values in the data
df.isna().mean().sort_values(ascending=False)


# In[28]:


df.head()


# # Representing gender with 0-1

# In[29]:


# change gender to 0-1 binary
df.loc[:, "gender"] = df.gender - 1


# In[30]:


# reduce interval in cholesterol & glucose from 1-3 to 0-2
df.loc[:, "cholesterol"] = df.cholesterol - 1
df.loc[:, "glucose"] = df.glucose - 1


# # Data Exploration

# In[31]:


num_cols = ["age", "bp_hi", "bp_lo"]


# In[32]:


for col in num_cols:
    sns.violinplot(x="disease", y=col, data=df)


# # BP Value Error

# In[33]:


# extreme values in bp_hi need to be corrected
bp_cols = ["bp_hi", "bp_lo"]
for col in bp_cols:
    sns.violinplot(x="disease", y=col, data=df)


# In[34]:


# 993 samples with extreme values for bp_hi or bp_lo
idx = df[(abs(df.bp_hi) > 300) | (abs(df.bp_lo) > 200)].index
df = df.drop(index=idx)


# In[35]:


# drop samples with negative bp_values
idx = df[(df.bp_hi < 0) | (df.bp_lo < 0)].index
df = df.drop(index=idx)


# In[36]:


# drop samples with bp_hi or bp_lo values less than 50; data entry error
idx = df[(df.bp_lo < 50) | (df.bp_hi < 50)].index
df = df.drop(index=idx)


# # High value Error

# In[37]:


# create column for height in ft
df["height_ft"] = df.height / 30.48

# drop samples with heights below 5 feet and above 7 feet
idx = df[(df.height_ft < 4.5) | (df.height_ft > 7)].index
df = df.drop(index=idx)


# # Features

# In[38]:


# blood pressure difference column
df["bp_diff"] = df.bp_hi - df.bp_lo

# BMI column to replace height and weight
# bmi = weight (kgs) / (height (m))^2
df["bmi"] = df.weight / (df.height / 100) ** 2

# added some more common measurement unit columns for better understanding
df["yrs"] = df.age / 365
df["height_ft"] = df.height / 30.48
df["weight_lbs"] = df.weight * 2.205


# In[39]:


# extreme values in bp_hi need to be corrected
bp_cols = ["bp_diff", "bmi", "height_ft", "weight_lbs"]
for col in bp_cols:
    sns.violinplot(x="disease", y=col, data=df)


# # After dropping errors the shape of data will be

# In[40]:


df.shape[0]


# In[41]:


feat = df["weight"]
feat1 = df[df.disease == 1]["weight"]
feat0 = df[df.disease == 0]["weight"]
fig, ax = plt.subplots()

ax.set_xlabel("weight")
ax.set_title(f"{'weight'} Distribution")
ax.legend()


# In[42]:


import plotly.figure_factory as ff

# Group data together
hist_data = [feat1, feat0]

group_labels = ["Disease", "No Disease"]

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels)
fig.update_layout(title_text=f"{'weight'} Distribution", xaxis_title="weight")


# In[43]:


feat = df["cholesterol"]
feat1 = df[df.disease == 1]["cholesterol"].value_counts()
feat0 = df[df.disease == 0]["cholesterol"].value_counts()
#     d = {'no_disease':feat0, 'disease':feat1}
#     f = pd.DataFrame(data=d)
fig, ax = plt.subplots()
#     st.write(f.style.background_gradient())
# fig, ax = plt.subplots()
width = 0.25
cd = ax.bar(
    x=feat1.index - width / 2,
    height=feat1,
    width=width,
    color="#e60909",
    label="Disease",
)
no_cd = ax.bar(x=feat0.index + width / 2,
    height=feat0,
    width=width,
    color="#09e648",
    label="No Disease",
)

# Attach a text label above each bar in *rects*, displaying its height
for rect in cd:
    height = rect.get_height()
    ax.annotate(
        "{}".format(height),
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha="center",
        va="bottom",
    )

for rect in no_cd:
    height = rect.get_height()
    ax.annotate(
        "{}".format(height),
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha="center",
        va="bottom",
    )

ax.set_xlabel("cholesterol")
ax.set_xticks(feat.unique())
ax.set_title("cholesterol")
ax.legend()
fig.tight_layout()


# In[44]:


import plotly.graph_objects as go

stat = "Cholesterol"
intervals = list(feat.unique())

fig = go.Figure(
    data=[
        go.Bar(name="Disease", x=intervals, y=list(feat1)),
        go.Bar(name="No Disease", x=intervals, y=list(feat0)),
    ]
)
# Change the bar mode
fig.update_layout(
    barmode="group", title_text=f"{stat} Distribution", xaxis_title=f"{stat} Values"
)


# In[45]:


sns.catplot(x="gender", y="bp_hi", hue="disease", kind="violin", split=True, data=df)
sns.catplot(x="gender", y="bp_lo", hue="disease", kind="violin", split=True, data=df)
sns.catplot(x="gender", y="bp_diff", hue="disease", kind="violin", split=True, data=df)


# In[46]:


import plotly.graph_objects as go


fig = go.Figure()

fig.add_trace(
    go.Violin(
        x=df["gender"][df["disease"] == 0],  # no disease
        y=df["bp_hi"][df["disease"] == 0],
        legendgroup="No Disease",
        scalegroup="Yes",
        name="No Disease",
        side="negative",
        line_color="#09e648",
    )
)
fig.add_trace(
    go.Violin(
        x=df["gender"][df["disease"] == 1],
        y=df["bp_hi"][df["disease"] == 1],
        legendgroup="Disease",
        scalegroup="Disease",
        name="Disease",
        side="positive",
        line_color="#e60909",
    )
)
fig.update_traces(meanline_visible=True)
fig.update_layout(violingap=0, violinmode="overlay")


# # Now Applying Machine learning Model

# # What is Gradient Boosting?
#
# Predictive analytics play an important role in clinical research. If a clinical condition or outcome can be accurately predicted, interventions can be delivered to the patient population who will benefit the most.
# Thus, numerous data mining algorithms have been developed for clinical prediction in nearly all subspecialties. Machine learning methods can be used in this setting as we can rely on the predictive algorithm to discover the
# nonlinearity and interaction structure in the data instead of hoping that the investigators will be wise enough to include such structure explicitly in their models.
#
# Among the many machine learning techniques, gradient boosting is a particularly attractive approach.In the clinical literature, gradient boosting has been successfully used to predict, among other things, cardiovascular events (2), development of sepsis (3), delirium (4) and hospital readmissions etc.

# # Gradient Boosting Model

# In[47]:


df.head()


# # Drop unnecessary columns

# In[48]:


drop_cols = [
    "disease",
    "yrs",
    "height_ft",
    "bp_diff",
    "weight_lbs",
    #"bmi",
    #"height",
    "weight",
]

X = df.drop(columns=drop_cols)
y = df.disease

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=28, stratify=df.disease
)


# In[49]:


# categorical columns to be encoded
cat_cols = ["cholesterol", "glucose"]
drop_cat = [0, 0]
# data preprocessing
preprocessing = ColumnTransformer(
    [
        #("encode_cats", OneHotEncoder(drop=drop_cat), cat_cols),
        #("encode_cats", LeaveOneOutEncoder(), cat_cols),
    ],
    remainder="passthrough",
)


# In[50]:


pipeline = Pipeline(
    [
        ("processing", preprocessing),
        ("model", XGBClassifier(use_label_encoder=False)),
    ]
)

pipeline.fit(X_train, y_train)

train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)




# # Fitting the model

# In[51]:


grid = {
    "model__n_estimators": np.arange(1, 3),
    "model__learning_rate": np.arange(0, 50, 10),
    #     "model__subsample": [],
    "model__colsample_bytree": np.arange(0.7,1,0.1),
    "model__max_depth": np.arange(4,7),
}
# fmt: on
pipeline_cv = GridSearchCV(pipeline, grid, cv=2, verbose=2, n_jobs=-1)
pipeline_cv.fit(X_train, y_train)

best_params = pipeline_cv.best_params_
best_params


# In[52]:


train_score = pipeline_cv.score(X_train, y_train)
test_score = pipeline_cv.score(X_test, y_test)


# # Feature importance

# In[53]:


feature_importances = pipeline_cv.best_estimator_["model"].feature_importances_
feature_importances = pd.DataFrame(
    {"feature": X_train.columns, "importance": feature_importances}
).sort_values("importance", ascending=False)
feature_importances


# # Predicting

# In[54]:


y_preds = pipeline_cv.predict(X_test)
preds_df, fig = predictions_df(X_test, y_test, y_preds)

# confusion matrix
cm = confusion_matrix(y_test, y_preds)

# classification report
class_report = classification_report(y_test, y_preds)

# prediction probabilities
pred_prob = pipeline_cv.predict_proba(X_test)
# add prediction probs to preds_df
preds_df["pred_prob"] = pred_prob[:, 1]

preds_df = preds_df.drop(columns=["residuals", "abs_residuals"])


# In[55]:


prob_true, prob_pred = calibration_curve(y_test, pred_prob[:, 1], n_bins=10)
plt.plot(prob_pred, prob_true, "-o")


# # Now predicting the accuracy with ROC(Receiver Operating Characteristic Curve )

# In[56]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_preds)


# # Result:
# An ROC curve can be considered as the average value of the sensitivity for a test over all possible values of specificity or vice versa.
# In general, an AUC of 0.5 suggests no discrimination (i.e., ability to diagnose patients with and without the disease or condition based on the test), 0.7 to 0.8 is considered acceptable, 0.8 to 0.9 is considered excellent, and more than 0.9 is considered outstanding.
#
# So here we are getting the accuracy rate is 0.7 that is acceptable.

# In[57]:


df


# In[58]:


df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
bmi= df['bmi']
bmi


# In[59]:


user_data = df[['bmi', 'age', 'gender', 'height', 'bp_hi', 'bp_lo', 'cholesterol', 'smoke']]
b = np.array(user_data, dtype=float) #  convert using numpy
c = [float(i) for i in user_data]


# In[ ]:


df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
gender = df['gender']
gender


# In[60]:


df['height'] = pd.to_numeric(df['height'], errors='coerce')
height = df['height']
height


# In[61]:


df['bp_lo'] = pd.to_numeric(df['bp_lo'], errors='coerce')
bp_lo = df['bp_lo']
bp_lo


# In[62]:


df['bp_hi'] = pd.to_numeric(df['bp_hi'], errors='coerce')
bp_hi = df['bp_hi']
bp_hi


# In[63]:


df['cholesterol'] = pd.to_numeric(df['cholesterol'], errors='coerce')
cholesterol = df['cholesterol']
cholesterol


# In[64]:


df['smoke'] = pd.to_numeric(df['smoke'], errors='coerce')
smoke = df['smoke']
smoke


# In[65]:


type(smoke)


# In[66]:


type(cholesterol)


# In[68]:


smoking_status = input("Enter smoking status (never, former, current): ")

if smoking_status == "never":
    numeric_value = 1
elif smoking_status == "former":
    numeric_value = 0
elif smoking_status == "current":
    numeric_value = -1
else:
    print("INvalid value")


# In[69]:


df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
bmi= df['bmi']
bmi

df['height'] = pd.to_numeric(df['height'], errors='coerce')
height = df['height']
height

df['cholesterol'] = pd.to_numeric(df['cholesterol'], errors='coerce')
cholesterol = df['cholesterol']
cholesterol

df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
gender = df['gender']
gender

df['bp_lo'] = pd.to_numeric(df['bp_lo'], errors='coerce')
bp_lo = df['bp_lo']
bp_lo

df['bp_hi'] = pd.to_numeric(df['bp_hi'], errors='coerce')
bp_hi = df['bp_hi']
bp_hi

df['smoke'] = pd.to_numeric(df['smoke'], errors='coerce')
smoke = df['smoke']
smoke


# In[70]:


type(smoking_status)


# In[71]:


x = df.drop('disease', axis=1)
x
y = df['disease']
y


# In[72]:


X = df.iloc[:, :7].values


# In[73]:


df



def comparison(weight,age,gender,bp_low,bp_high,smoking):
    # Create a list to store the input data
    user_data = [weight, age, cholesterol, gender, bp_low, bp_high, smoking]

    # In[75]:

    from sklearn.linear_model import LogisticRegression

    # Initialize the logistic regression model
    log_reg = LogisticRegression()

    # Train the model on the input features and output labels
    log_reg.fit(X, y)

    # In[76]:

    type(weight)

    # In[77]:

    type(age)

    # In[78]:

    type(cholesterol)

    # In[79]:

    type(bp_low)

    # In[80]:

    type(bp_high)

    # In[81]:

    type(smoking)

    # In[82]:

    type(user_data)

    # In[83]:

    # Use the trained model to make a prediction on the user's input data
    prediction = log_reg.predict([user_data])

    # Print the prediction
    print("Prediction: ", prediction)

    # In[84]:

    # Use the trained model to make a prediction on the user's input data
    prediction_proba = log_reg.predict_proba([user_data])[0][1]
    # Print the predicted likelihood of getting the disease
    return "{:.2f}%".format(prediction_proba * 100)



__all__ = ["comparison"]


# In[ ]:




