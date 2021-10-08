# Importing Necessary Libraries.
import pandas as pd
import re
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler,StandardScaler, Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.metrics import r2_score

import math    
# import geopandas as gpd #pip
# import adjustText as aT #pip
from sklearn.ensemble import GradientBoostingRegressor
   
from sklearn.svm import SVR
#pip install adjustText
#Clean the Column names


def clean_col_names(df):
    try:
        df.columns= [re.sub(r'[^A-Za-z]',' ', x).strip().lower() for x in df.columns]
        df.columns= ['_'.join(x.split()) for x in df.columns]
        index= df.dtypes.values.tolist().index('datetime64[ns]')
        df.columns.values[index] = "date"
    except:
        pass
    
def missing_values(df, start_index= 0,  strategy= 'mean'):
    from sklearn.impute  import SimpleImputer
    imp_mean = SimpleImputer(missing_values = np.nan, strategy = strategy)
    df.iloc[:,start_index:] = imp_mean.fit_transform(df.iloc[:, start_index:])
    return df

def missing_values_mode(df):
    cols= df.columns[2:]
    for col in cols:
        df[col]= df[col].fillna(df[col].mode()[0])
    return df

def make_education(df):
    try:
        clean_col_names(education_bor)
        df = df[df.year.isin([2018,2019,2020])]
        df.area = df.area.str.replace(' and', ' &')
        df = df[df.area.str.lower().isin(list(crime_month_la.ocu_name.str.lower().unique()))]
        clean_df = pd.DataFrame(columns = df.columns[1:3])
        for cols in df.qualifications_of_working_age_population.unique():
            cols = cols.lower().strip()
            clean_df[cols] = ''
        for rows in df.iterrows():
            X =clean_df.index[(clean_df.area== rows[1].area) & (clean_df.year == rows[1].year)]
            if(len(X) == 0):
               clean_df =  clean_df.append({clean_df.columns[0]: rows[1][1], 
                                 clean_df.columns[1]: rows[1][2], 
                                 rows[1][3].strip().lower():rows[1].percent,
                                 }, ignore_index = True)
            else:
                clean_df.loc[X, rows[1][3].strip().lower()] = rows[1].percent
        clean_df.year = pd.to_datetime(clean_df.year, format='%Y')
        clean_df=clean_df.replace(['!','#','*'], np.nan)
    except Exception as e:
        print(e)
        
    return clean_df

def filter_unemployment(data):
    d18 = data.iloc[1:34]
    d19 = data.iloc[79:112]
    d20 = data.iloc[157:190]
    data = pd.concat([d18,d19,d20])    
    data.columns = [ele[0].lower() for ele in data.columns.str.split('.')]
    data = data.drop(columns= 'conf')
    data = data.replace(['!','#','*'], np.nan).reset_index().drop(columns='index')
    data = make_unemployment(data) 
    data = data.infer_objects()
    return data 

def make_ethnicity(df):
    try:
        df = df[df.age  == 'All ages']
        df = df.loc[:, ['borough','ethnic_group',2018,2019,2020]].reset_index().drop(columns= 'index')
        clean_cols = list(['borough','year'])
        for col in df.ethnic_group.unique():
            clean_cols.insert(2, col.strip().lower())
        clean_df = pd.DataFrame(columns=clean_cols)
        for rows in df.iterrows():
            X = clean_df[clean_df.borough == rows[1].iloc[0]]
            if(len(X)==0):
                for i in [2018,2019,2020]:
                    clean_df = clean_df.append({clean_df.columns[0]: rows[1].iloc[0], 
                                    clean_df.columns[1]: i, 
                                    rows[1].iloc[1].lower().strip(): rows[1][i] , 
                                    }, ignore_index = True)
            else:
                for i,j in zip([2018,2019,2020],X.index):
                    clean_df.loc[j][rows[1].iloc[1].lower().strip()] = rows[1][i]
        #clean_col_names(clean_df)
        clean_df.borough = clean_df.borough.str.replace(' and', ' &')
        clean_df= clean_df.rename({'borough':'area', 'year': 'date'},axis=1)
        clean_df['pop_black']= clean_df[['black caribbean', 'black african', 'white & black caribbean']].sum(axis=1)
        clean_df['pop_white']= clean_df[['other white', 'white irish', 'white british']].sum(axis=1)
        clean_df['pop_other_mixed(arab,etc)']= clean_df[['other mixed','arab']].sum(axis=1)
        clean_df['pop_asian']= clean_df[['chinese','bangladeshi', 'pakistani','indian']].sum(axis=1)
        clean_df['pop_mixed_asian']= clean_df[['white & asian', 'other asian']].sum(axis=1)
        clean_df.pop_black = pd.to_numeric(clean_df.pop_black)
        clean_df = clean_df[['area','date','pop_black','pop_white',
                             'pop_mixed_asian', 'pop_asian', 
                             'pop_other_mixed(arab,etc)', 'all persons', 'bame']].rename(columns  ={'all persons': 'pop_total', 'bame': 'pop_bame'})    
        clean_df = clean_df.infer_objects()
    except Exception as e:
        print("Error Exception In Make_Ethnicity {}".format(e))
    
    return clean_df        

def make_unemployment(df):
    
    df.area = df.area.apply(lambda x: str(x).split(':')[-1])
    df.area = df.area.str.replace(' and', ' &')
    df = df.loc[:, df.columns.str.contains('area|rate')]
    df = imputer(df,1 )
    date = []
    for idx in enumerate(df.index):
        if(idx[0] < 33):
            date.append(2018)
        elif(idx[0]>32 and idx[0]<=65):
            date.append(2019)
        else:
            date.append(2020)
    df['date'] = date
    return df

def imputer(df, start_index= 0):
    imp= KNNImputer(n_neighbors = 5)
    df.iloc[:, start_index:] = imp.fit_transform(df.iloc[:, start_index:]).round(2)
    return df   


def make_low_income(df):
    df = df.iloc[10:, :6]
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.infer_objects()
    df = df.groupby(by= 'Borough').sum()
    df = df.reset_index().rename(columns  = {'Borough':'area', '2017/18':2018, '2018/19':2019, '2019/20':2020 })
    df['area']= df['area'].str.replace(' and', ' &')
    inc1 = df[['area', 2018]].rename(columns = {2018: 'children_lowincome'} )
    inc1['year'] = 2018
    inc2 = df[['area', 2019]].rename(columns = {2019: 'children_lowincome'} )
    inc2['year'] = 2019
    inc3 = df[['area', 2020]].rename(columns = {2020: 'children_lowincome'} )
    inc3['year'] = 2020
    low_income = pd.concat([inc1, inc2, inc3],axis=0)
    
    return low_income

def make_inactive(inactive):
    df= pd.DataFrame()
    for data in inactive:
        eco_inactive = inactive[data]
        q= pd.concat([eco_inactive['Unnamed: 1'] , eco_inactive.loc[:, eco_inactive.columns[eco_inactive.columns.str.contains('%')]]], axis=1 ).rename(
            columns = {'Unnamed: 1': 'area'}).drop(0, axis=0).iloc[2:, :]
        q['year']= data
        q = q.replace(['!', '#','~'],np.nan )
        df = df.append(q)
    df.year = pd.to_datetime(df.year)
    df.area = df.area.str.replace(' and', ' &')
    df.columns = ['area', 'inactive_whites', 'inactive_minorities', 'inactive_Mixedethnicgroups','eco_inactive_indians','inactive_pakistani/bangla', 'inactive_black', 'inactive_otherethnicgroups', 'year']

    return df
    
def make_master(approach, daily=False,employment_overall= True):
    # Merge Crime_Month and Education
    merged = crime_month_la.rename({'ocu_name': 'area'}, axis=1).merge(education_bor.rename({'year':'date'},axis =1),
                                                                       how ='left',left_on=['area', crime_month_la['date'].dt.year],
                                                                       right_on =['area',education_bor.year.dt.year ]
                                                                       ).drop(columns= ['key_1', 'date_y']).rename({'date_x': 'date'} , axis=1)
        
    
    if(employment_overall == True):
        df = pd.read_excel('./Data/unemplyment_data/eco_inactive_timeseries_overall.xls')
        overall= df[df.year.str.contains('2018|2019|2020')].loc[454:]
        overall.year = pd.to_datetime(overall.year)
        merged = merged.merge(overall, how= 'left', left_on = merged.date, right_on = overall.year).drop(columns= ['key_0', 'year']).rename({'rate':'unemployed_count'},axis=1)
        
       
        
    
    else:
        merged = merged.merge(unemployment_la, how= 'left', left_on = ['area' , merged.date.dt.year],
                              right_on = ['area', unemployment_la.date]).drop(columns= ['key_1', 'date_y']).rename({'date_x': 'date'} , axis=1)
   
    merged = merged.merge(ethnic_bor,how = 'left', left_on=[merged.area, merged.date.dt.year],
                          right_on= [ethnic_bor.area, ethnic_bor.date.astype('int64')]).rename({'key_0':'area', 
                                    'date_x':'date'},
                                  axis=1).drop(columns=['key_1','area_x', 'area_y', 'date_y'])
    
    merged =merged.merge(child_protection, left_on=['area', merged.date.dt.year], 
                         right_on= ['area', 'date'], how= 'left').drop(columns= ['date','date_y']).rename(columns={'date_x':'date'})
    
    if(approach ==2):
        merged =merged.merge(low_income, left_on= ['area', merged.date.dt.year], right_on= ['area', 'year'], how= 'left').drop(columns='year')
        merged = merged.merge(eco_inactive, left_on= ['area',merged.date.dt.year], right_on= ['area', eco_inactive.year.dt.year]).drop(['key_1', 'year'], axis=1 )
        cols = merged.drop(['area','date'],axis=1).columns
        merged[cols] = merged[cols].apply(pd.to_numeric)
        
        sns.set_theme()
        sns.set(font_scale = 2)
        plt.figure(figsize=(30, 25))
        sns.set(font_scale = 1.6)
        s = sns.heatmap(merged.corr().round(2), linewidths =2, annot = True)
        plt.xticks(rotation = 90, fontsize= 20)
        plt.yticks(rotation = 0, fontsize=20)
        plt.savefig('./plots/{}_corr.png'.format(approach),dpi =300, bbox_inches = "tight")
        plt.clf
        
        
        
        
    elif(approach == 1):
        merged = merged.merge(eco_inactive_all,  left_on = ['area', merged.date.dt.year], right_on = ['area', 'year'], how= 'left')
        sns.set_theme()
        sns.set(font_scale = 2)
        plt.figure(figsize=(30, 25))
        sns.set(font_scale = 2)
        s = sns.heatmap(merged.drop(columns= 'year').corr().round(2), linewidths =2, annot = True)
        plt.xticks(rotation = 90, fontsize= 20)
        plt.yticks(rotation = 0, fontsize=20)
        plt.savefig('./plots/{}_corr.png'.format(approach),dpi =300, bbox_inches = "tight")
        plt.clf
        
    return merged
def prepare_for_regression(df, approach):
    if(approach  ==2):    
        df= one_hot_month(df)
        df = pd.concat([pd.get_dummies(df.area, drop_first=True), pd.get_dummies(df.date, drop_first = True),df.loc[:, 'nvq4+':], df.knife_crime_offs], axis=1 )#.drop(columns = 'year')
        df= missing_values(df, start_index=2, strategy = 'mean')
    elif(approach ==1):
        df = one_hot_month(df)
        df= pd.concat([pd.get_dummies(df.area, drop_first=True), 
                      pd.get_dummies(df.date, drop_first = True),
                      df.loc[:, 'nvq4+':], df.knife_crime_offs], axis=1 )
        df= imputer(df)
   
    return df 
    
    
    
def weekly_knife_crime_plot(df):
    mappings= {0:"Mon",1:"Tue",2:"Wed", 3:"Thu", 4:"Fri",5:"Sat", 6:"Sun"}
    df['Day'] = df.date.dt.weekday.apply(lambda x : mappings[x])   
    df= df.groupby(by= 'Day').knife_crime_offs.mean()
    df = df.loc[[i for i in mappings.values()]]
    sns.lineplot(data = df)
    plt.xlabel('Days of the Week')
    plt.ylabel('Knife Crime Offs')    
    plt.title('Daily Knife Crime Offences | Averaged Over Year 2018-2020')
    plt.savefig('./plots/Daily_average_Knife_crime.png')

def bar_plot_all_crimes(df):
    grouped = df.groupby([df.date.dt.year]).sum()
    grouped = grouped.drop(columns = 'tno_offs')
    #Plotting
    years = [2018,2019,2020]
    for year in years:
        # f = plt.figure(figsize=(30,20))
        sns.barplot(x = list(grouped.columns), y= list(grouped[grouped.index == year].iloc[0]))
        plt.xticks(rotation =75)
        plt.xlabel('Offenses')
        plt.ylabel('Count')
        plt.title('Crime offs year - {}'.format(year))
        plt.savefig('./plots/Crime_{}.png'.format(year))
    f = plt.figure(figsize=(20,10))
    sns.lineplot(data = grouped.T)
    plt.xticks(rotation= 75)
    plt.xlabel('Offense')
    plt.ylabel('Count')
    plt.title('Comparison of Years 2018-2020')    
    
def top10_crime_pie_chart(df):
    years = [2018,2019,2020]
    for year in years:
        data = df[df.date.dt.year == year]
        data = data.sum()
        explode = (0.1, 0,0,0,0,0, 0,0,0,0)
        fig1, ax1 = plt.subplots(figsize=(25,20))
        data =  data.drop(['tno_offs', 'total_knife_crime_offs','ocu_name']).sort_values(ascending  =False).head(10)
        ax1.pie(x =data.values, autopct='%1.1f%%',
                textprops={'fontsize': 18},
                shadow=True, startangle=180, pctdistance = 0.7 ,explode = explode)
        ax1.axis('equal')
        plt.title('Top - 10 Criminal Offenses London for year {}'.format(year), fontdict={'fontsize': 25})
        plt.legend(data.index.to_list(),prop={'size':20}, loc=3)

def import_data():
    global crime_month_la, master, ethnic_bor, unemployment_la, education_bor, low_income, eco_inactive, child_protection ,eco_inactive_all
    crime_master_list = []    
    crime_month_la = pd.read_excel('./Data/crime_data/Borough Monthly 2018-20.xlsx')
    crime_master_list.append(crime_month_la)
    #Cleaning the Crime Dataset
    for data in crime_master_list:
        data = clean_col_names(data)
        del(data)

    #Replacing Missing values with Mode
    # crime_month_la = missing_values(df = crime_month_la, start_index= 2, strategy = 'mean')
    #Creating New Column Merging all Knife Crimes into One.
    
    crime_month_la= crime_month_la[['ocu_name','date', 'knife_crime_offs']]
    #bar_plot_all_crimes(crime_month_la)   
    
    
    # Unemployment Data
    unemployment_la = pd.read_csv('./Data/unemplyment_data/unemployment_2018_2020_borough.csv')
    unemployment_la = filter_unemployment(unemployment_la)
    unemployment_la['unemployed_youth'] = unemployment_la.iloc[:, 1:5].sum(axis=1) 
    unemployment_la = unemployment_la[['area','date', 'unemployed_youth']]
    
    # Education Data
    education_bor = pd.read_csv('./Data/Qualification Data Borough/Qualifications-of-working-age-NVQ.csv')
    education_bor = make_education(education_bor)
    
    # Ethnicity_population 2011 census
    ethnic_bor = pd.read_excel('./Data/population_ethnic/Ethnic group projections (2016-based housing-led).xlsx','Population - Persons')
    ethnic_bor = make_ethnicity(ethnic_bor)
    
    # Childrens in Low income households 
    low_income = pd.read_excel('./Data/poverty/Children in low income families.xlsx',sheet_name = 1)
    low_income = make_low_income(low_income)
    
    # Economically Inactive- Working age population
    eco_inactive= pd.read_excel('./Data/economic_inactive/economic-inactivity-by-ethnic.xlsx', sheet_name = ['2018','2019'])
    eco_inactive = make_inactive(eco_inactive)
    
    # Economically Inactive 2018 -2020 Borough
    eco_inactive_all = pd.read_csv('./Data/economic_inactive/economic-inactivity.csv').dropna()
    eco_inactive_all = pd.concat([eco_inactive_all.Area, eco_inactive_all.iloc[:, -10: ]], axis = 1)
    eco_inactive_all = eco_inactive_all.iloc[:, eco_inactive_all.columns.str.contains('Area|percent')]
    eco_inactive_all = make_eco_inactive_all(eco_inactive_all)
    
    # Children in Child Protection Plan
    child_protection = pd.read_excel('./Data/Child protection/child_protection_plan.xls', sheet_name = 1)
    child_protection = make_child_protection(child_protection)
    
    
def make_eco_inactive_all(df):
    df.columns = ['area', 2018,2019,2020]
    df.area = df.area.str.replace(' and', ' &')
    df =df.replace('!', np.nan)
    inc1 = df.iloc[:, [0,1]].rename(columns = {2018:'Workingage_EcoInactive'})
    inc1['year'] = 2018
    inc2 = df.iloc[:, [0,2]].rename(columns = {2019:'Workingage_EcoInactive'})
    inc2['year'] = 2019
    inc3 = df.iloc[:, [0,3]].rename(columns = {2020:'Workingage_EcoInactive'})
    inc3['year'] = 2020
    df = pd.concat([inc1, inc2, inc3],axis=0)
    df.iloc[:, 1]  = df.iloc[:, 1].astype(float)
    return df


def make_child_protection(df):
    df.columns = df.iloc[1]
    df = df.iloc[3:, 1:18].replace('x', np.nan).rename(columns = {'Area':
                                                                  'area'})
    df = df.fillna(df.mean())
    q= pd.DataFrame()
    for i in [2018, 2019, 2020]:
        temp = df[['area',i]].rename(columns= {i: 'CPP_claimants'})
        temp['date']= i
        q = q.append(temp, ignore_index = True)        
    q.area= q.area.str.replace(' and', ' &')
    return q

def convert_percent_to_values():
    percent_cols = (master.loc[:, master.columns.str.contains('rate')]).columns
    percent_cols = percent_cols.append(education_bor.columns[2:])
    for col in percent_cols:
        master[col] = master[col]* (master['all persons']/100)

def one_hot_month(df):
    month = {1: "Jan",2: "Feb",3: "Mar",4: "Apr", 5: "May", 6: "June", 7: "July",
             8:"Aug", 9:"Sep",10: "Oct", 11: "Nov", 12: "Dec"}
    df['date'] =[month[i] for i in df.date.dt.month]
    
    return df


def borough_maps(df, column_name):
    # set the filepath and load in a shapefile


    df= df.groupby(['area',df.date.dt.year])['children_lowincome'].mean()
    df = df.groupby('area').sum()
    # df = df.groupby(['ocu_name', df.date.dt.year]).sum()
    # df = df.groupby('ocu_name').mean()
    path = './Data/Geo_Boundaries/statistical-gis-boundaries-london/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp'
    mdf = gpd.read_file(path)
    mdf.NAME= mdf.NAME.str.replace(' and', ' &')
    mdf = mdf.set_index('NAME')
    map_df = mdf.join(df[column_name])
    fig, ax = plt.subplots(1, figsize=(15, 10))
    # create map
    map_df.plot(edgecolor = 'grey', ax= ax , column = column_name, cmap= 'Reds')        
    ax.axis('off')
    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=map_df[column_name].min(), vmax=map_df[column_name].max()))
    # empty array for the data range
    sm._A = []
    # add the colorbar to the figure
    
    cbar = fig.colorbar(sm)
    plt.title('Something')
    if('knife' in column_name):
        ax.set_title('Knife crimes in London- {}'.format(year), fontdict={'fontsize': '30', 'fontweight' :'4'}
                 ,loc= 'center' )
    else:
        ax.set_title('{} in London- {}'.format(column_name,year), fontdict={'fontsize': '30', 'fontweight' :'4'}
                 ,loc= 'center' )
    # Add Labels
    map_df['coords'] = map_df['geometry'].apply(lambda x: x.representative_point().coords[:])
    map_df['coords'] = [coords[0] for coords in map_df['coords']]
    for idx, row in map_df.iterrows():
        plt.annotate(s=row.name, xy=row['coords'], horizontalalignment='center')
    plt.savefig('./plots/{}_{}.png'.format(column_name,year))
    plt.close()
    
def rfe_feature_selection(model, X,y):
    global col_names
    rfe = RFE(estimator=model)
    rfe.fit(X, y)
    col_names = X.loc[:,rfe.get_support()].columns
    selected_features = rfe.transform(X)
    sc = StandardScaler()
    selected_features = sc.fit_transform(selected_features)
    return selected_features,y


def overall_inactive_london():
    df = pd.read_excel('./Data/unemplyment_data/eco_inactive_timeseries_overall.xls')
    inactive = df.iloc[454:490].reset_index().drop(columns= 'index')
    temp =master[master.date.dt.year ==2018]



# feature selection
def feature_selection(X, y, minmax=0):
    from sklearn.feature_selection import f_regression, SelectKBest
    global col_names
    fs = SelectKBest(score_func=f_regression, k = 10)
    fs.fit(X, y)
    col_names = X.loc[:,fs.get_support()].columns
    X_selected = fs.transform(X)
    if(minmax ==1):
        sc = MinMaxScaler()
    else:
        sc = StandardScaler()
    X_selected = sc.fit_transform(X_selected)

    return X_selected,y

def split_train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state = 123)    
    
    return X_train, X_test,y_train,y_test 

def boxknife(df):
    sns.boxplot(data = df['knife_crime_offs'], palette= 'Set2', orient = 'v') 
    plt.xticks(rotation = 75)
    plt.xlabel('Knife Crime Offences')
    plt.ylabel('Count')
    plt.title('Outliers in Crime Data')
    plt.savefig('./plots/outliers.png',bbox_inches = 'tight' ,dpi =200)
    plt.clf()
    
def boxknife_cleaned(df):
    sns.boxplot(data = df['knife_crime_offs'], palette= 'Set2', orient = 'v') 
    plt.xticks(rotation = 75)
    plt.xlabel('Knife Crime Offences')
    plt.ylabel('Count')
    plt.title('Outliers Eliminated in Crime Data')
    plt.savefig('./plots/outliers_eliminated.png',bbox_inches="tight",  dpi =200)
    plt.clf()
    
def outlier_removal(df):
    boxknife(df)
    q1= df.knife_crime_offs.quantile(0.25)
    q3= df.knife_crime_offs.quantile(0.75)
    iqr= q3- q1 
    lower_limit = q1 - 1.5*iqr
    upper_limit = q3  + 1.5*iqr    
    df = df[~((df.knife_crime_offs < lower_limit) |(df.knife_crime_offs >upper_limit))]
    
    return df

    
def regression_model(clf,X,y):
    import shap
    X_train, X_test , y_train, y_test= split_train_test(X, y)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse ,rmse, r2 = actual_predicted_plot(clf, y_test, y_pred)
    if('Linear' in str(clf)):
        plot_feature_importance(clf.coef_,clf)
    else: 
        plot_feature_importance(clf.feature_importances_ ,clf )
    
    return mse, rmse ,r2

def plot_feature_importance(importance,clf):
    sns.set_theme()
    # summarize feature importance
    # for i,v in zip(col_names, importance):
    	# print('Feature:'+str()+'Score:'+str(v))
    # plot feature importance
    df = pd.DataFrame({'col': col_names , 'imp': importance})
    df = df.sort_values(by='imp', ascending=False)
    df = df.head(30)
    plt.figure(figsize = (7,4))
    plt.barh(df['col'].astype(str),df['imp'])
    plt.ylabel('Features')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Plot')
    plt.xticks(rotation= 90)
    plt.savefig('./plots/FeatureImportance_{}_{}.png'.format(clf,approach),bbox_inches = "tight", dpi =300)
    
    plt.clf()
    
def actual_predicted_plot(clf, y_test, y_pred):
    if('Gradient' in str(clf)):
        clf = "Gradient Boosting Regressor"
    elif('Random' in str(clf)):
        clf = "Random Forest Regressor"
    elif('Linear' in str(clf)):
        clf = "Linear Regression"
    else:
        clf = "Decision Trees"
    df= pd.DataFrame({'Actual': y_test.astype(int), 'Predicted': y_pred.astype(int)})
    sns.set_theme()
    df = df.sort_values('Actual')
    sns.lineplot(data= df, palette = 'tab10')
    plt.title('Actual vs Predicted Plot')
    mse = mean_squared_error(y_test.astype(int), y_pred.astype(int))
    rmse = math.sqrt(mean_squared_error(y_test.astype(int), y_pred.astype(int)))
    r2 =      r2_score(y_test, y_pred)
    print(clf)
    print("MSE: {}".format(round(mse,3) ))
    print("RMSE: {} ".format(round(rmse, 3)))
    print("R2 Error: {}".format(round(r2, 5)))
    plt.savefig('./plots/Actual_Predicted_{}_{}.png'.format(clf,approach), dpi =300)
    plt.clf
    return mse , rmse, r2

def corr_heatmap(df):
    df = crime_month_la
    sns.set_theme()
    sns.set(font_scale = 2)
    columns = abs(df.corr().knife_crime_offs) > 0.40
    df =  df[columns[columns].index]
    plt.figure(figsize=(18, 16))
    s = sns.heatmap(df.corr(), annot = True,cmap = sns.cm.rocket_r)
    plt.xticks(rotation = 90, fontsize= 20)
    plt.yticks(rotation = 0, fontsize=20)
    plt.savefig('./plots/corr_sub.png',dpi =300, bbox_inches = "tight")
    plt.clf()

def timeseriesplt_knife(df):
    sns.set_theme()
    sns.set(font_scale = 2)
    plt.figure(figsize= (13,8))
    df.groupby('date')['knife_crime_offs'].sum().plot()
    plt.ylabel('Knife Crime Offences (count)')
    plt.title('Time Series Plot (Knife Crime Offences) 2018-2020')
    plt.savefig('./plots/Knife_crime_2018_2020.png',bbox_inches= "tight", dpi =300)
    plt.clf
    
def ethnicity_plt(df):
    df = df.mean()[1:].drop('pop_total').sort_values()
    explode = (0, 0,0,0,0, 0.1)
    sns.set(font_scale = 2)
    plt.figure(figsize= (13,8))
    plt.pie(x =df.values, autopct='%1.1f%%',
            textprops={'fontsize': 18},
            shadow=False, startangle=0, pctdistance = 0.7 ,explode = explode)
    plt.axis('equal')
    plt.title('Population by Ethnicity - 2018-2020 Average', fontdict={'fontsize': 25})
    plt.legend(['Other Mixed(Arab,etc)','Mixed Asians','Black','Asians','BAME','Whites'],prop={'size':15}, loc= 3)
    plt.savefig('./plots/Ethnicity_percent.png', bbox_inches="tight",dpi =300)
    plt.clf
    
    
def heatmapinactive(df):
    sns.set_theme()
    sns.set(font_scale = 2) 
    plt.figure(figsize=(18, 16))
    sns.heatmap(df.groupby('area').mean()[eco_inactive.columns[1:-1]])
    plt.title("Borough level Heatmap of Economically Inactive Ethnic Groups", fontsize= 40)
    plt.xticks(rotation = 90, fontsize= 25)
    plt.yticks(rotation = 0, fontsize=25)
    
    
def masterheatmap(df):
    msc = MinMaxScaler()
    # msc= StandardScaler()
    df= df.groupby(['area',df.date.dt.year]).mean()
    df= df.groupby('area').mean().drop(['knife_crime_offs', 'unemployed_count'],axis=1)
    df = pd.concat([df, crime_month_la.groupby('ocu_name')['knife_crime_offs'].sum()],axis=1)
    d = msc.fit_transform(df.loc[:, 'nvq4+':])
    q = pd.DataFrame(d,columns=df.loc[:, 'nvq4+':].columns)
    df = pd.concat([df.reset_index()['index'], q], axis = 1)
    df = df.sort_values(by = 'knife_crime_offs', ascending = False)
    
   
    df = df.set_index('index')
    sns.set_theme()
    sns.set(font_scale = 3) 
    plt.figure(figsize=(25, 25))
    sns.heatmap(df, cmap="Blues")
    plt.ylabel('Borough Names')
    plt.xlabel('Features')
    plt.title('Heatmap of Socio-Economic Factors Across London', fontsize=35)
    plt.savefig('./plots/heatmap_socio_eco.png',dpi = 300, bbox_inches = 'tight')
    plt.clf
    
    sns.set_theme()
    sns.set(font_scale = 2) 
    plt.figure(figsize=(13, 5))
    sns.heatmap(df.iloc[0:10,:], cmap="Blues")
    plt.ylabel('Borough Names')
    plt.xlabel('Features')
    plt.title('Knife Crime Analysis - Top-10 Boroughs')    
    plt.savefig('./plots/analysis_10.png', bbox_inches = "tight", dpi =200)
    plt.clf
    
    
    sns.set_theme()
    sns.set(font_scale = 2) 
    plt.figure(figsize=(13, 5))
    sns.heatmap(df.iloc[len(df) - 10: len(df),:], cmap="Blues")
    plt.ylabel('Borough Names')
    plt.xlabel('Features')
    plt.title('Knife Crime Analysis - Bottom-10 Boroughs')
    plt.savefig('./plots/analysis_b_10.png' , bbox_inches = "tight", dpi =200)
    plt.clf
    
def top_knife_crime(df):
    sns.set_theme()
    sns.set_theme(font_scale=2)
    plt.figure(figsize=(10,15))
    df= df.groupby(['ocu_name',df.date.dt.year]).sum().reset_index()
    df = df.groupby('ocu_name').mean().knife_crime_offs.reset_index().sort_values('knife_crime_offs')
    # df= crime_month_la.groupby('ocu_name').sum().sort_values('knife_crime_offs', ascending =False).reset_index()
    plt.barh(df.ocu_name, df.knife_crime_offs)
    plt.ylabel('Boroughs')
    plt.xlabel('Knife Crime Count')
    # plt.gca().invert_yaxis()
    plt.title('Knife Crime Offences  - Averaged (2018-2020)',fontsize = 25)
    plt.savefig('./plots/barplot_borough_knife.png', bbox_inches = 'tight', dpi =300)


    
def main():
    # os.chdir('C:/Users/piyus/Desktop/Final project')
    import_data()
    global approach
    for approach in [1,2]:
        master = make_master(approach, daily = False, employment_overall= True)
        #Clean Master Data File 
        master = missing_values(master, 2)
        # Plots. 
        #______________________________________________
        masterheatmap(master)
        master = prepare_for_regression(master, approach)
        master= outlier_removal(master)
        boxknife_cleaned(master)
        ethnicity_plt(ethnic_bor)
        top_knife_crime(crime_month_la)
        corr_heatmap(master)
        timeseriesplt_knife(crime_month_la)
        #______________________________________________
        
        X = master.drop(columns = ['knife_crime_offs'])
        y = master.knife_crime_offs
        models= [LinearRegression(), 
                 DecisionTreeRegressor(random_state=1),  
                 RandomForestRegressor(random_state = 1), 
                 GradientBoostingRegressor(learning_rate = 0.08 ,random_state = 1),
                 ]
        Scores = pd.DataFrame(columns = ['Model', 'MSE', 'RMSE', 'R2'])
        for model in models:
            a,b = feature_selection(X,y, minmax = 0)
      
            mse, rmse, r2 = regression_model(model
                             ,a,b)
            print("\n")
            
            Scores= Scores.append({'Model':model, 'MSE':mse, 'RMSE':rmse, 'R2':r2}, ignore_index = True)
        Scores.to_csv('./Modelresults_approach_{}.csv'.format(approach))
        best = Scores.sort_values(by = 'R2').iloc[-1].Model
        print("Best Performing Model (Approach {}): {} \n \n".format(approach, best))


if __name__ == "__main__":
    main()
    
