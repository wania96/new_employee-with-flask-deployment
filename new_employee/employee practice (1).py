#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[31]:


df=pd.read_csv(r"C:\Users\Haier\Downloads\deployment\Employee_Dataset_new_noise.csv")


# In[32]:


df.head()


# In[33]:


df.shape


# In[34]:


df.isnull().sum()


# In[35]:


df.fillna('nan')
from sklearn.impute import SimpleImputer
mode_imputer=SimpleImputer(missing_values=np.nan , strategy='most_frequent')
for i in df.columns:
    
    df[i]=pd.DataFrame(mode_imputer.fit_transform(df[[i]]))
    


# In[36]:


df.isnull().sum()


# In[37]:


df.dtypes


# In[38]:


df=df.drop(columns={'Unnamed: 0','Employee_ID',"Employee_Name",'Hire Date'},axis=1)


# In[39]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['Gender']=lb.fit_transform(df['Gender'])
df[' MaritalStatus']=lb.fit_transform(df[' MaritalStatus'])
df['Department']=lb.fit_transform(df['Department'])
df['Position']=lb.fit_transform(df['Position'])
df['Education']=lb.fit_transform(df['Education'])
df['EnvironmentSatisfaction']=lb.fit_transform(df['EnvironmentSatisfaction'])
df['JobInvolvement']=lb.fit_transform(df["JobInvolvement"])
df['JobSatisfaction']=lb.fit_transform(df['JobSatisfaction'])
df['RelationshipSatisfaction']=lb.fit_transform(df['RelationshipSatisfaction'])
df['WorkLifeBalance']=lb.fit_transform(df['WorkLifeBalance'])
df['Behaviourial_Competence']=lb.fit_transform(df['Behaviourial_Competence'])
df['OntimeDelivery']=lb.fit_transform(df['OntimeDelivery'])
df['TicketSolvingManagements']=lb.fit_transform(df['TicketSolvingManagements'])
df['Working_from_home']=lb.fit_transform(df['Working_from_home'])
df['Psycho-social_indicators']=lb.fit_transform(df['Psycho-social_indicators'])
df['Netconnectivity']=lb.fit_transform(df['Netconnectivity'])


# In[40]:


df['PerformanceRating']=lb.fit_transform(df['PerformanceRating'])


# In[41]:


df.dtypes


# In[42]:


df.columns


# In[43]:


sns.boxplot(df['Age'])


# In[44]:


df.rename(columns = {' MaritalStatus':'MaritalStatus', 'Psycho-social_indicators':'Psycho_social_indicators'}, inplace = True)


# In[45]:


df.dtypes


# In[46]:


df['Age'] = pd.to_numeric(df['Age'])


# In[ ]:





# In[47]:


import statsmodels.formula.api as smf
formula='PerformanceRating~Age+Gender+MaritalStatus+Department+Position+Education+EnvironmentSatisfaction+JobInvolvement+JobLevel+JobSatisfaction+Annual_Income+RelationshipSatisfaction+Working_hrs_perday+TotalWorkingYearsexperience+TrainingTimeinmonths+WorkLifeBalance+Behaviourial_Competence+OntimeDelivery+TicketSolvingManagements+Project_evlaution+Working_from_home+Psycho_social_indicators+overtime+PercentSalaryHike+Netconnectivity'
model=smf.ols(formula,data=df).fit()


# In[48]:


print(model.summary2())


# In[49]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
xs=df.drop(['PerformanceRating'],axis=1)
vif_data = pd.DataFrame()
vif_data["VIF"] = [variance_inflation_factor(xs.values, i)
                          for i in range(len(xs.columns))]
vif_data["feature"] = xs.columns
vif_data.round(1)


# In[50]:


x1=xs.drop(['Working_hrs_perday'],axis=1)


# In[51]:


vif_data = pd.DataFrame()
vif_data["VIF"] = [variance_inflation_factor(x1.values, i)
                          for i in range(len(x1.columns))]
vif_data["feature"] = x1.columns
vif_data.round(1)


# In[52]:


x2=x1.drop(['Annual_Income'],axis=1)


# In[53]:


vif_data = pd.DataFrame()
vif_data["VIF"] = [variance_inflation_factor(x2.values, i)
                          for i in range(len(x2.columns))]
vif_data["feature"] = x2.columns
vif_data.round(1)


# In[54]:


x3=x2.drop(['Netconnectivity'],axis=1)


# In[55]:


vif_data = pd.DataFrame()
vif_data["VIF"] = [variance_inflation_factor(x3.values, i)
                          for i in range(len(x3.columns))]
vif_data["feature"] = x3.columns
vif_data.round(1)


# In[56]:


x4=x3.drop(['Department'],axis=1)


# In[57]:


vif_data = pd.DataFrame()
vif_data["VIF"] = [variance_inflation_factor(x4.values, i)
                          for i in range(len(x4.columns))]
vif_data["feature"] = x4.columns
vif_data.round(1)


# In[68]:


x5=x4.drop(['Age'],axis=1)


# In[69]:


vif_data = pd.DataFrame()
vif_data["VIF"] = [variance_inflation_factor(x5.values, i)
                          for i in range(len(x5.columns))]
vif_data["feature"] = x5.columns
vif_data.round(1)


# In[71]:


x6=x5.drop(['Project_evlaution'],axis=1)


# In[72]:


vif_data = pd.DataFrame()
vif_data["VIF"] = [variance_inflation_factor(x6.values, i)
                          for i in range(len(x6.columns))]
vif_data["feature"] = x6.columns
vif_data.round(1)


# In[73]:


x6.shape


# In[74]:


y=df['PerformanceRating']
x=x6


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)


# In[77]:


from sklearn import metrics 
from sklearn.metrics import classification_report,accuracy_score


# In[78]:


from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression()
model_log.fit(X_train, y_train)
y_pred = model_log.predict(X_test)
print ("Accuracy : ", metrics.accuracy_score(y_test, y_pred))


# from sklearn.ensemble import RandomForestClassifier
# 
# rf_clf = RandomForestClassifier(n_estimators=100)
# model=rf_clf.fit(X_train, y_train)
# y_test_pred=model.predict(X_test)
# print('accuracy',accuracy_score(y_test, y_test_pred))

# importance = rf_clf.feature_importances_
# importance=np.sort(importance)
# # summarize feature importance
# for i,v in enumerate(importance):
#     print('Feature: {}, Score: {}'.format(i,v))
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
#  
# # making predictions on the testing set
# y_pred = gnb.predict(X_test)
#  
# # comparing actual response values (y_test) with predicted response values (y_pred)
# from sklearn import metrics
# print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

# In[ ]:


import pickle


# In[286]:


model=pickle.dumps(model_log)


# In[287]:


f=open(r'C:\Users\Haier\new_employee\model.pickle','wb')


f.write(model)


f.close()

