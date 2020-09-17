import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as pl
from matplotlib.colors import LinearSegmentedColormap
from catboost import CatBoostRegressor
from PIL import Image

import plotly.offline as po
import plotly.graph_objs as go
import plotly.express as px

import plotly.offline as py
import plotly.figure_factory as ff
from plotly.offline import iplot
#import cufflinks
#cufflinks.go_offline()
# Set global theme
#cufflinks.set_config_file(world_readable=True, theme='pearl')

matplotlib.use('Agg')

#path='C:/Users/clark.djilo/Desktop/CLV/'


## Loads the CLV data
df_shapley = pd.read_csv('Data_Auto_imputed.csv', sep=';',decimal=",",encoding = "ISO-8859-1",engine='python')


st.write(""" # Exploration Of The Historical Data""")
st.write('---')


st.write(""" ## 1. Correlation Matrix""")
st.write('---')

image_corr = Image.open('Correlation.PNG')
st.image(image_corr)

st.write('**Conclusions:**')
st.write('* The variables of the data set are not highly correlated')

# # one hot encoding the remaining categorical variables 
# df_correlation = pd.get_dummies(df_shapley,columns=['Coverage',
#                                 'Education',
#                                 'EmploymentStatus',
#                                 'Gender',
#                                 'Zone',
#                                 'Marital.Status',
#                                 'Policy.Type',
#                                 'Policy',
#                                 'Renew.Offer.Type',
#                                 'Sales.Channel',
#                                 'Vehicle.Class',
#                                 'Vehicle.Size'])

# # separating the dependent and independent variables
# df_correlation = df_correlation.drop(['Group.Clustering','Ratio.Clv.Prime_Tot','Clv'],1)
# colorscales = [
#     'Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu', 'Reds', 'Blues',
#     'Picnic', 'Rainbow', 'Portland', 'Jet', 'Hot', 'Blackbody', 'Earth',
#     'Electric', 'Viridis', 'Cividis'
# ]
# corrs = df_correlation.corr()

# figure = ff.create_annotated_heatmap(
#     z=corrs.values,
#     x=list(corrs.columns),
#     y=list(corrs.index),
#     colorscale='Blues',
#     annotation_text=corrs.round(1).values,
#     showscale=True, reversescale=True)

# figure.layout.margin = dict(l=100, t=100)
# figure.layout.height = 1600
# figure.layout.width = 2000

# if st.button('Press to see the Correlation Matrix please'):
#     #iplot(figure)
#     #st.write(figure)
#     st.plotly_chart(figure)
    
#     #st.write('Please look at the new opened tab to see the correlation matrix')
#     st.write('**Conclusions:**')
#     st.write('* The variables of of the data set are not correlated')




st.write(""" ## 2. Clustering""")
st.write('---')

st.write(""" ## 2.1. Principles""")

st.write('The clustering is a non supervised method wchich allows to create similar groups from a dataset')



st.write(""" ## 2.2. Conclusions of the study""")

st.write('The clustering allows to create 4 groups and we are going to deepdive in these groups to characterize them')



st.write(""" ## 2.3. Characterization of the groups""")

st.write('The clustering allows to create 4 groups and we are going to deepdive in these groups to characterize them')


st.write(""" ### 2.3.1 Groups VS Categorical Data""")

group_cat = st.selectbox('GROUP_C',['Group.1',
                        'Group.2',
                        'Group.3',
                        'Group.4'
                        ])

x_cat =  st.selectbox('XC_CAT',['Coverage',
                        'Education',
                        'EmploymentStatus',
                        'Gender',
                        'Zone',
                        'Marital.Status',
                        'Policy.Type',
                        'Policy',
                        'Renew.Offer.Type',
                        'Sales.Channel',
                        'Vehicle.Class',
                        'Vehicle.Size'
                        ])

y_cat =  st.selectbox('YC_CAT',['Clv','Months.Since.Policy.Inception','Number.of.Policies','Total.Claim.Amount','Ratio.Clv.Prime_Tot'])

#df_shapley_cat = df_shapley[df_shapley.Group.Clustering == group_cat]

df_shapley_cat = df_shapley.loc[df_shapley['Group.Clustering'] == group_cat]

pivot = pd.pivot_table(df_shapley_cat,index=[x_cat],values=[y_cat],aggfunc= np.mean)
pivot[y_cat] = pivot.mean(axis=1).round(4)
pivot = pd.DataFrame(pivot.to_records())

fig1 = px.bar(pivot, x_cat, y_cat)
#fig1.show()
st.write(fig1)
#barplot_chart = st.plotly_chart(fig)







st.write(""" ### 2.3.2 Groups VS Numerical Data""")


group_num = st.selectbox('GROUP_N',['Group.1',
                        'Group.2',
                        'Group.3',
                        'Group.4'
                        ])


x_num =  st.selectbox('XN_NUM',['Months.Since.Policy.Inception',
                                'Income',
                                'Location.Geo',
                                'Location.Code',
                                'Monthly.Premium.Auto',
                                'Months.Since.Last.Claim',
                                'Number.of.Open.Complaints'
                                ])


y_num =  st.selectbox('YN_NUM',['Clv','Number.of.Policies','Total.Claim.Amount','Ratio.Clv.Prime_Tot'])

df_shapley_num = df_shapley.loc[df_shapley['Group.Clustering'] == group_num]

fig2 = px.scatter_matrix(df_shapley_num[[x_num,y_num]])
#fig2.show()
st.write(fig2)
#scatter_chart = st.plotly_chart(fig)








# df_clust = df_shapley.drop(['Group.Clustering','Ratio.Clv.Prime_Tot','Clv'],1)

# st.write(""" ## 2.1. Snapshot of the data before the clustering - List of the first 5 records""")
# st.write(df_clust.head(5))




# st.write(""" ## 2.2. Clustering Study""")
# image_hospital = Image.open('ClusterPlot.png')
# st.image(image_hospital)

# st.write('**Conclusions:**')
# st.write('* The algorithm allows to create 4 groups and link each record to his group')



# st.write(""" ## 2.3. Snapshot at the data after the clustering""")
# st.write('Below is a snapshot is listed of the first 5 records from the created dataframe')

# df_clust_after = df_shapley.drop(['Ratio.Clv.Prime_Tot','Clv'],1)

# st.write(df_clust_after.head(5))

# st.write('**Conclusions:**')
# st.write('* We can see in the dataframe bellow the group allocated to each records')


st.write(""" ## 3. Exploration of the categorical data""")
st.write('Select a X_CAT categorical data and cross it with the Y_CAT selected variable')
st.write('---')

x_cat =  st.selectbox('X_CAT',['Coverage',
                        'Education',
                        'EmploymentStatus',
                        'Gender',
                        'Zone',
                        'Marital.Status',
                        'Policy.Type',
                        'Policy',
                        'Renew.Offer.Type',
                        'Sales.Channel',
                        'Vehicle.Class',
                        'Vehicle.Size',
                        'Group.Clustering'
                        ])

y_cat =  st.selectbox('Y_CAT',['Clv','Months.Since.Policy.Inception','Number.of.Policies','Total.Claim.Amount','Ratio.Clv.Prime_Tot'])


pivot = pd.pivot_table(df_shapley,index=[x_cat],values=[y_cat],aggfunc= np.mean)
pivot[y_cat] = pivot.mean(axis=1).round(4)
pivot = pd.DataFrame(pivot.to_records())

#pivot

fig1 = px.bar(pivot, x_cat, y_cat)
#fig1.show()
st.write(fig1)
#barplot_chart = st.plotly_chart(fig)

st.write(""" ## 4. Exploration of the numerical data""")
st.write('Select a X_NUM numerical data and cross it with the Y_NUM selected variable')
st.write('---')

x_num =  st.selectbox('X_NUM',['Months.Since.Policy.Inception',
                                'Income',
                                'Location.Geo',
                                'Location.Code',
                                'Monthly.Premium.Auto',
                                'Months.Since.Last.Claim',
                                'Number.of.Open.Complaints'
                                ])


y_num =  st.selectbox('Y_NUM',['Clv','Number.of.Policies','Total.Claim.Amount','Ratio.Clv.Prime_Tot'])


fig2 = px.scatter_matrix(df_shapley[[x_num,y_num]])
#fig2.show()
st.write(fig2)
#scatter_chart = st.plotly_chart(fig)






    
# x =  st.selectbox('X',['Months.Since.Policy.Inception',
#                         'Coverage',
#                         'Education',
#                         'EmploymentStatus',
#                         'Gender',
#                         'Income',
#                         'Location.Geo',
#                         'Location.Code',
#                         'Zone',
#                         'Marital.Status',
#                         'Monthly.Premium.Auto',
#                         'Months.Since.Last.Claim',
#                         'Number.of.Open.Complaints',
#                         'Number.of.Policies',
#                         'Policy.Type',
#                         'Policy',
#                         'Renew.Offer.Type',
#                         'Sales.Channel',
#                         'Total.Claim.Amount',
#                         'Vehicle.Class',
#                         'Vehicle.Size'
#                         ])
    
# y =  st.selectbox('Y',['Clv'])    
    
    

# typeOfFeature = str(df_shapley.dtypes[x])


# if typeOfFeature == 'object':
    
#     pivot = pd.pivot_table(df_shapley,index=[x],values=[y],aggfunc= np.mean)
#     pivot[y] = pivot.mean(axis=1).round(0)
#     pivot = pd.DataFrame(pivot.to_records())

#     #pivot

#     fig1 = px.bar(pivot, x, y)
#     #fig1.show()
#     st.write(fig1)
#     #barplot_chart = st.plotly_chart(fig)
    
# else:
    
#     fig2 = px.scatter_matrix(df_shapley[[x,y]])
#     #fig2.show()
#     st.write(fig2)
#     #scatter_chart = st.plotly_chart(fig)
    
    

st.write(""" # New Client Lifetime Value Prediction""")
st.write('In this part we will predict the Client Lifetime Value of a new client based on machine learning models')
st.write('---')


# Best Model

df_shapley = df_shapley.drop(['Group.Clustering','Ratio.Clv.Prime_Tot'],1)

# one hot encoding the remaining categorical variables 
df_shapley = pd.get_dummies(df_shapley,columns=['Coverage',
                                'Education',
                                'EmploymentStatus',
                                'Gender',
                                'Zone',
                                'Marital.Status',
                                'Policy.Type',
                                'Policy',
                                'Renew.Offer.Type',
                                'Sales.Channel',
                                'Vehicle.Class',
                                'Vehicle.Size'])

# separating the dependent and independent variables
X_train = df_shapley.drop('Clv',1)
y_train = df_shapley['Clv']

# Need to load JS vis in the notebook
shap.initjs()

#xgb_model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001, random_state=0)

xgb_model =   CatBoostRegressor(iterations = 1000,
                                learning_rate = 0.03,
                                depth = 2,
                                l2_leaf_reg = 1,
                                loss_function = 'RMSE',
                                border_count = 50,
                                silent = True,
                                random_state = 5009)

xgb_model.fit(X_train, y_train)










# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('New Client Parameters')

def user_input_features():
    
    CUSTOMER_ID  =  st.sidebar.number_input('CustomerID',min_value=0,max_value=10000,value=0,step=1)
    COVERAGE =  st.sidebar.selectbox('Coverage',['Basic','Extended','Premium'])
    EDUCATION =  st.sidebar.selectbox('Education',['Bachelor','College','Master','High School or Below','Doctor'])
    EMPLOYMENTSTATUS =  st.sidebar.selectbox('EmploymentStatus',['Employed','Disabled','Medical Leave','Unemployed'])
    GENDER =  st.sidebar.selectbox('Gender',['M','F'])
    INCOME =  st.sidebar.slider('Income',min_value=0.0,max_value=100000.0,value=25000.0)  #st.sidebar.slider('Income',X.Income.min(), X.Income.max(), X.Income.mean())
    LOCATION_GEO =  st.sidebar.slider('Location.Geo',min_value=12.0,max_value=30.3,value=15.0,step=0.1)
    LOCATION_CODE =  st.sidebar.slider('Location.Code',min_value=70.0,max_value=90.7,value=20.0,step=0.1)
    ZONE =  st.sidebar.selectbox('Zone',['Rural','Urban','Suburban'])
    MARITAL_STATUS =  st.sidebar.selectbox('Marital.Status',['Single','Married','Divorced'])
    MONTHLY_PREMIUM_AUTO =  st.sidebar.number_input('Monthly.Premium.Auto',min_value=60,max_value=300,value=100,step=1)
    MONTHS_SINCE_LAST_CLAIM =  st.sidebar.number_input('Months.Since.Last.Claim',min_value=0,max_value=35,value=0,step=1)
    MONTHS_SINCE_POLICY_INCEPTION =  st.sidebar.number_input('Months.Since.Policy.Inception',min_value=0,max_value=99,value=0,step=1)
    NUMBER_OF_OPEN_COMPLAINTS =  st.sidebar.number_input('Number.of.Open.Complaints',min_value=0,max_value=5,value=0,step=1)
    NUMBER_OF_POLICIES =  st.sidebar.number_input('Number.of.Policies',min_value=0,max_value=9,value=0,step=1)
    POLICY_TYPE =  st.sidebar.selectbox('Policy.Type',['Personal Auto','Special Auto','Corporate Auto'])
    POLICY =  st.sidebar.selectbox('Policy',['Personal L1','Special L2','Corporate L1','Corporate L2','Personal L3','Personal L2','Special L1','Special L3','Corporate L3'])
    RENEW_OFFER_TYPE =  st.sidebar.selectbox('Renew.Offer.Type',['Offer1','Offer2','Offer4','Offer3'])
    SALES_CHANNEL =  st.sidebar.selectbox('Sales.Channel',['Agent','Branch','Call Center','Web'])
    TOTAL_CLAIM_AMOUNT =  st.sidebar.slider('Total.Claim.Amount',min_value=0.0,max_value=3000.0,value=500.0)
    VEHICLE_CLASS =  st.sidebar.selectbox('Vehicle.Class',['Four-Door Car','Luxury SUV','Two-Door Car','SUV','Luxury Car','Sports Car'])
    VEHICLE_SIZE =  st.sidebar.selectbox('Vehicle.Size',['Medsize','Large','Small'])

    data = {'CustomerID': CUSTOMER_ID,
            'Coverage': COVERAGE,
            'Education': EDUCATION,
            'EmploymentStatus': EMPLOYMENTSTATUS,
            'Gender': GENDER,
            'Income': INCOME,
            'Location.Geo': LOCATION_GEO,
            'Location.Code': LOCATION_CODE,
            'Zone': ZONE,
            'Marital.Status': MARITAL_STATUS,
            'Monthly.Premium.Auto': MONTHLY_PREMIUM_AUTO,
            'Months.Since.Last.Claim': MONTHS_SINCE_LAST_CLAIM,
            'Months.Since.Policy.Inception': MONTHS_SINCE_POLICY_INCEPTION,
            'Number.of.Open.Complaints': NUMBER_OF_OPEN_COMPLAINTS,
            'Number.of.Policies': NUMBER_OF_POLICIES,
            'Policy.Type': POLICY_TYPE,
            'Policy': POLICY,
            'Renew.Offer.Type': RENEW_OFFER_TYPE,
            'Sales.Channel': SALES_CHANNEL,
            'Total.Claim.Amount': TOTAL_CLAIM_AMOUNT,
            'Vehicle.Class': VEHICLE_CLASS,
            'Vehicle.Size': VEHICLE_SIZE}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# one hot encoding the remaining categorical variables 
columns = ['CustomerID',
'Income',
'Location.Geo',
'Location.Code',
'Monthly.Premium.Auto',
'Months.Since.Last.Claim',
'Months.Since.Policy.Inception',
'Number.of.Open.Complaints',
'Number.of.Policies',
'Total.Claim.Amount',
'Coverage_Basic',
'Coverage_Extended',
'Coverage_Premium',
'Education_Bachelor',
'Education_College',
'Education_Doctor',
'Education_High School or Below',
'Education_Master',
'EmploymentStatus_Disabled',
'EmploymentStatus_Employed',
'EmploymentStatus_Medical Leave',
'EmploymentStatus_Unemployed',
'Gender_F',
'Gender_M',
'Zone_Rural',
'Zone_Suburban',
'Zone_Urban',
'Marital.Status_Divorced',
'Marital.Status_Married',
'Marital.Status_Single',
'Policy.Type_Corporate Auto',
'Policy.Type_Personal Auto',
'Policy.Type_Special Auto',
'Policy_Corporate L1',
'Policy_Corporate L2',
'Policy_Corporate L3',
'Policy_Personal L1',
'Policy_Personal L2',
'Policy_Personal L3',
'Policy_Special L1',
'Policy_Special L2',
'Policy_Special L3',
'Renew.Offer.Type_Offer1',
'Renew.Offer.Type_Offer2',
'Renew.Offer.Type_Offer3',
'Renew.Offer.Type_Offer4',
'Sales.Channel_Agent',
'Sales.Channel_Branch',
'Sales.Channel_Call Center',
'Sales.Channel_Web',
'Vehicle.Class_Four-Door Car',
'Vehicle.Class_Luxury Car',
'Vehicle.Class_Luxury SUV',
'Vehicle.Class_SUV',
'Vehicle.Class_Sports Car',
'Vehicle.Class_Two-Door Car',
'Vehicle.Size_Large',
'Vehicle.Size_Medsize',
'Vehicle.Size_Small'
]

df_dummy = pd.DataFrame(index=[0],columns=columns) 

df_dummy['CustomerID']=df['CustomerID']
df_dummy['Income']=df['Income']
df_dummy['Location.Geo']=df['Location.Geo']
df_dummy['Location.Code']=df['Location.Code']
df_dummy['Monthly.Premium.Auto']=df['Monthly.Premium.Auto']
df_dummy['Months.Since.Last.Claim']=df['Months.Since.Last.Claim']
df_dummy['Months.Since.Policy.Inception']=df['Months.Since.Policy.Inception']
df_dummy['Number.of.Open.Complaints']=df['Number.of.Open.Complaints']
df_dummy['Number.of.Policies']=df['Number.of.Policies']
df_dummy['Total.Claim.Amount']=df['Total.Claim.Amount']



if df['Coverage'].iat[0]=='Basic':
   df_dummy['Coverage_Basic']=1
else:
   df_dummy['Coverage_Basic']=0 


if df['Coverage'].iat[0]=='Extended':
    df_dummy['Coverage_Extended']=1 
else: 
    df_dummy['Coverage_Extended']=0


if df['Coverage'].iat[0]=='Premium':
    df_dummy['Coverage_Premium']=1 
else: 
    df_dummy['Coverage_Premium']=0

if df['Education'].iat[0]=='Bachelor':
    df_dummy['Education_Bachelor']=1 
else: 
    df_dummy['Education_Bachelor']=0


if df['Education'].iat[0]=='College':
    df_dummy['Education_College']=1 
else: 
    df_dummy['Education_College']=0


if df['Education'].iat[0]=='Doctor':
    df_dummy['Education_Doctor']=1 
else: 
    df_dummy['Education_Doctor']=0


if df['Education'].iat[0]=='High School or Below':
    df_dummy['Education_High School or Below']=1 
else: 
    df_dummy['Education_High School or Below']=0


if df['Education'].iat[0]=='Master':
    df_dummy['Education_Master']=1 
else: 
    df_dummy['Education_Master']=0


if df['EmploymentStatus'].iat[0]=='Disabled':
    df_dummy['EmploymentStatus_Disabled']=1 
else: 
    df_dummy['EmploymentStatus_Disabled']=0


if df['EmploymentStatus'].iat[0]=='Employed':
    df_dummy['EmploymentStatus_Employed']=1 
else: 
    df_dummy['EmploymentStatus_Employed']=0


if df['EmploymentStatus'].iat[0]=='Medical Leave':
    df_dummy['EmploymentStatus_Medical Leave']=1 
else: 
    df_dummy['EmploymentStatus_Medical Leave']=0


if df['EmploymentStatus'].iat[0]=='Unemployed':
    df_dummy['EmploymentStatus_Unemployed']=1 
else: 
    df_dummy['EmploymentStatus_Unemployed']=0


if df['Gender'].iat[0]=='F':
    df_dummy['Gender_F']=1 
else: 
    df_dummy['Gender_F']=0

if df['Gender'].iat[0]=='M':
    df_dummy['Gender_M']=1 
else: 
    df_dummy['Gender_M']=0


if df['Zone'].iat[0]=='Rural':
    df_dummy['Zone_Rural']=1 
else: 
    df_dummy['Zone_Rural']=0


if df['Zone'].iat[0]=='Suburban':
    df_dummy['Zone_Suburban']=1 
else: 
    df_dummy['Zone_Suburban']=0


if df['Zone'].iat[0]=='Urban':
    df_dummy['Zone_Urban']=1 
else: 
    df_dummy['Zone_Urban']=0


if df['Marital.Status'].iat[0]=='Divorced':
    df_dummy['Marital.Status_Divorced']=1 
else: 
    df_dummy['Marital.Status_Divorced']=0


if df['Marital.Status'].iat[0]=='Married':
    df_dummy['Marital.Status_Married']=1 
else: 
    df_dummy['Marital.Status_Married']=0


if df['Marital.Status'].iat[0]=='Single':
    df_dummy['Marital.Status_Single']=1 
else: 
    df_dummy['Marital.Status_Single']=0


if df['Policy.Type'].iat[0]=='Corporate Auto':
    df_dummy['Policy.Type_Corporate Auto']=1 
else: 
    df_dummy['Policy.Type_Corporate Auto']=0


if df['Policy.Type'].iat[0]=='Personal Auto':
    df_dummy['Policy.Type_Personal Auto']=1 
else: 
    df_dummy['Policy.Type_Personal Auto']=0


if df['Policy.Type'].iat[0]=='Special Auto':
    df_dummy['Policy.Type_Special Auto']=1 
else: 
    df_dummy['Policy.Type_Special Auto']=0


if df['Policy'].iat[0]=='Corporate L1':
    df_dummy['Policy_Corporate L1']=1 
else: 
    df_dummy['Policy_Corporate L1']=0


if df['Policy'].iat[0]=='Corporate L2':
    df_dummy['Policy_Corporate L2']=1 
else: 
    df_dummy['Policy_Corporate L2']=0


if df['Policy'].iat[0]=='Corporate L3':
    df_dummy['Policy_Corporate L3']=1 
else: 
    df_dummy['Policy_Corporate L3']=0


if df['Policy'].iat[0]=='Personal L1':
    df_dummy['Policy_Personal L1']=1 
else: 
    df_dummy['Policy_Personal L1']=0


if df['Policy'].iat[0]=='Personal L2':
    df_dummy['Policy_Personal L2']=1 
else: 
    df_dummy['Policy_Personal L2']=0


if df['Policy'].iat[0]=='Personal L3':
    df_dummy['Policy_Personal L3']=1 
else: 
    df_dummy['Policy_Personal L3']=0


if df['Policy'].iat[0]=='Special L1':
    df_dummy['Policy_Special L1']=1 
else: 
    df_dummy['Policy_Special L1']=0


if df['Policy'].iat[0]=='Special L2':
    df_dummy['Policy_Special L2']=1 
else: 
    df_dummy['Policy_Special L2']=0


if df['Policy'].iat[0]=='Special L3':
    df_dummy['Policy_Special L3']=1 
else: 
    df_dummy['Policy_Special L3']=0


if df['Renew.Offer.Type'].iat[0]=='Offer1':
    df_dummy['Renew.Offer.Type_Offer1']=1 
else: 
    df_dummy['Renew.Offer.Type_Offer1']=0


if df['Renew.Offer.Type'].iat[0]=='Offer2':
    df_dummy['Renew.Offer.Type_Offer2']=1 
else: 
    df_dummy['Renew.Offer.Type_Offer2']=0


if df['Renew.Offer.Type'].iat[0]=='Offer3':
    df_dummy['Renew.Offer.Type_Offer3']=1 
else: 
    df_dummy['Renew.Offer.Type_Offer3']=0


if df['Renew.Offer.Type'].iat[0]=='Offer4':
    df_dummy['Renew.Offer.Type_Offer4']=1 
else: 
    df_dummy['Renew.Offer.Type_Offer4']=0


if df['Sales.Channel'].iat[0]=='Agent':
    df_dummy['Sales.Channel_Agent']=1 
else: 
    df_dummy['Sales.Channel_Agent']=0


if df['Sales.Channel'].iat[0]=='Branch':
    df_dummy['Sales.Channel_Branch']=1 
else: 
    df_dummy['Sales.Channel_Branch']=0


if df['Sales.Channel'].iat[0]=='Call Center':
    df_dummy['Sales.Channel_Call Center']=1 
else: 
    df_dummy['Sales.Channel_Call Center']=0


if df['Sales.Channel'].iat[0]=='Web':
    df_dummy['Sales.Channel_Web']=1 
else: 
    df_dummy['Sales.Channel_Web']=0


if df['Vehicle.Class'].iat[0]=='Four-Door Car':
    df_dummy['Vehicle.Class_Four-Door Car']=1 
else: 
    df_dummy['Vehicle.Class_Four-Door Car']=0


if df['Vehicle.Class'].iat[0]=='Luxury Car':
    df_dummy['Vehicle.Class_Luxury Car']=1 
else: 
    df_dummy['Vehicle.Class_Luxury Car']=0


if df['Vehicle.Class'].iat[0]=='Luxury SUV':
    df_dummy['Vehicle.Class_Luxury SUV']=1 
else: 
    df_dummy['Vehicle.Class_Luxury SUV']=0


if df['Vehicle.Class'].iat[0]=='SUV':
    df_dummy['Vehicle.Class_SUV']=1 
else: 
    df_dummy['Vehicle.Class_SUV']=0


if df['Vehicle.Class'].iat[0]=='Sports Car':
    df_dummy['Vehicle.Class_Sports Car']=1 
else: 
    df_dummy['Vehicle.Class_Sports Car']=0

if df['Vehicle.Class'].iat[0]=='Two-Door Car':
    df_dummy['Vehicle.Class_Two-Door Car']=1 
else: 
    df_dummy['Vehicle.Class_Two-Door Car']=0


if df['Vehicle.Size'].iat[0]=='Large':
    df_dummy['Vehicle.Size_Large']=1 
else: 
    df_dummy['Vehicle.Size_Large']=0


if df['Vehicle.Size'].iat[0]=='Medsize':
    df_dummy['Vehicle.Size_Medsize']=1 
else: 
    df_dummy['Vehicle.Size_Medsize']=0


if df['Vehicle.Size'].iat[0]=='Small':
    df_dummy['Vehicle.Size_Small']=1 
else: 
    df_dummy['Vehicle.Size_Small']=0


#df_dummy



# Main Panel

# Print specified input parameters
st.write(' ## 1.Competion between machine learning models')
st.write('This table presents the result of the competition of regression models on our data')
st.write('---')

image_compare = Image.open('CompareModel.PNG')
st.image(image_compare)

st.write('**Conclusions:**')

st.write('* The CatBoost model performs best among the 22 models put into competition with the best indicators (MAE, MSE, ...). It will be our Best Model')

st.write('* The performance of the LinearRegression model is not good.It ranks 15th out of the 22 models put in competition')

# Print specified input parameters
#st.write(' ## 1.Specified Input parameters')
#st.write('These are the parameters of the new client define by the user in the sidebar')
#st.write(df)
#st.write('---')




# Prediction for the best model

st.write(' ## 2.Predicted Value Of The Client Value Lifetime - **Best Model**')
st.write('---')

st.write('New client parameters:')
st.write(df)

new_prediction = xgb_model.predict(df_dummy)

new_prediction = str(round(new_prediction[0],2)) + ' €'

st.success('The prediction for the **Best Model** of the new client is {}'.format(new_prediction))








## Shapley Value for the best model

st.write(' ## 3.Interpretation of the prediction for the **Best Model**')
st.write('---')

st.write('This is the interpretation of the prediction for the new client')

explainer = shap.TreeExplainer(xgb_model)


shap_values_0 = explainer.shap_values(df_dummy)    

i = 0
shap.force_plot(explainer.expected_value, shap_values_0[i], features=df_dummy.loc[i], feature_names=df_dummy.columns,matplotlib=True,show=False
            ,figsize=(16,5))
st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
pl.clf()




st.write('This is the global interpretation of the **Best Model** : The most importants variables for the Client Lifetime Value prediction')

shap_values_1 = explainer.shap_values(X_train)  

shap.summary_plot(shap_values_1, features=X_train, feature_names=X_train.columns,show=False) 
st.pyplot(bbox_inches='tight')
pl.clf()    

    
    
    
## Linear Regression
st.write('## 4.Predicted Value Of The Client Value Lifetime - **Linear Regression Model**')
st.write('---')


st.write('New client parameters:')
st.write(df)


from sklearn.linear_model import LinearRegression

lr_model =   LinearRegression(copy_X=True, fit_intercept=False, n_jobs=-1, normalize=True)

lr_model.fit(X_train, y_train)

new_prediction = lr_model.predict(df_dummy)

new_prediction = str(round(new_prediction[0],2)) + ' €'

st.success('The prediction for the **Linear Regression Model** of the new client is {}'.format(new_prediction))

#if st.button('4.Why this prediction ?'):
explainer = shap.LinearExplainer(lr_model,X_train,feature_dependence="independent")

shap_values_0 = explainer.shap_values(df_dummy)    


st.write('This is the interpretation of the prediction (**Linear Regression Model**) for the new client')

i = 0
shap.force_plot(explainer.expected_value, shap_values_0[i], features=df_dummy.loc[i], feature_names=df_dummy.columns,matplotlib=True,show=False
            ,figsize=(16,5))
st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
pl.clf()


st.write('This is the global interpretation of model (**Linear Regression Model**): The most importants variables for the Client Lifetime Value prediction')


shap_values_1 = explainer.shap_values(X_train)  

shap.summary_plot(shap_values_1, features=X_train, feature_names=X_train.columns,show=False) 
st.pyplot(bbox_inches='tight')
pl.clf()
