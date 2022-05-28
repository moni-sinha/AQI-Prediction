from django.shortcuts import render
from django.http import HttpResponse
import django 

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from io import BytesIO
import urllib, base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor

# Create your views here.

def home(request):
    return render(request, 'home.html', {'name':'moni'});

# def get_graph():
#     buffer = BytesIO()
#     plt.savefig(buffer, format='png')

#     buffer.seek(0)
#     image_png = buffer.getvalue()
#     graph = base64.b64encode(image_png)
#     graph = graph.decode('utf-8')
#     buffer.close()
#     return graph

def add(request):

    # plt.plot([1, 2, 3]) 
    # fig = get_graph()


    #val1 = request.POST['PM25']
    #val2 = request.POST['NO']
    #res = int(val1) + int(val2)


    df  = pd.read_csv("C:\\Users\\hp\\Downloads\\FinalChennai.csv")
    #df.head()



    df = df.drop(columns=['City', 'Date'])
    #df

    # checking for null values

    # df.isnull()
    #sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')


    X = df.iloc[:, :-2] #independent feature
    Y = df.iloc[:, -2] #dependent feature

    #sns.pairplot(df)

    #df.corr()

    corrmat = df.corr()
    top_corr_features = corrmat.index
    #plt.figure(figsize = (10, 5))
    g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')

    #corrmat.index


    # ## feature importance
    # 
    # feature importance gives you a score for each feature of your data, the higher the score, the more relevant is the feature towards your output variable.
    # 
    # For that, we use extra tree regressor


    
    model = ExtraTreesRegressor()
    model.fit(X, Y)


    #df.head()


    #print(model.feature_importances_)


    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(5).plot(kind='barh')  # 5 to extract best 5 features
    #plt.show()

    #sns.distplot(Y) # right skewed

    # train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


    from sklearn.ensemble import RandomForestRegressor
    regressor=RandomForestRegressor()
    regressor.fit(X_train,Y_train)


    #regressor.coef_

    #regressor.intercept_

    #print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, Y_train)))


    #print("Coefficient of determination R^2 <-- on test set: {}".format(regressor.score(X_test, Y_test)))

    from sklearn.model_selection import cross_val_score
    score=cross_val_score(regressor, X, Y, cv=5)

    #score.mean()

    prediction=regressor.predict(X_test)

    #sns.distplot(Y_test-prediction)


    #plt.scatter(Y_test,prediction)


    from sklearn import metrics

    #print('MAE:', metrics.mean_absolute_error(Y_test, prediction))
    #print('MSE:', metrics.mean_squared_error(Y_test, prediction))
    #print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, prediction)))


    # calculates the maximum residual error.
    #metrics.max_error(Y_test, prediction)

    #  (coefficient of determination) regression score function.

    # Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
    # A constant model that always predicts the expected value of y, disregarding the input features, 
    # would get a  score of 0.0.
    #metrics.r2_score(Y_test, prediction)

    df=df.rename(columns = {'PM2.5':'PM25'})

    a = request.POST['a']
    b = request.POST['b']
    c = request.POST['c']
    d = request.POST['d']
    e = request.POST['e']
    f = request.POST['f']
    g = request.POST['g']
    h = request.POST['h']
    i = request.POST['i']
    j = request.POST['j']

    newData = {'PM25':[a], 
           'NO':[b],
           'NO2':[c],
           'NOx':[d],
           'NH3':[e],
           'CO':[f],
           'SO2':[g],
           'O3':[h],
           'Benzene':[i],
           'Toluene':[j]}

    dftemp = pd.DataFrame(newData)
    #print(dftemp)
    aqi_value = regressor.predict(dftemp)
    #float(aqi_value)
    return render(request, 'result.html', {'result':aqi_value});
def aqi(request):
    return render(request, 'content.html');


def content(request):
    return render(request, 'content2.html');

def live_aqi(request):
    return render(request, 'live_aqi.html');

def new(request):
    return render(request, 'new.html');    

    