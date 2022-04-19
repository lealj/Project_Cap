import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_color
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
import statsmodels.api as sma


import seaborn as sns

regions = ['Latin America & Caribbean', 'South Asia', 'Sub-Saharan Africa', 'Europe & Central Asia', 'Middle East & North Africa', 'East Asia & Pacific', 'North America']
filenameGDP = r'GDP_Per_Capita.xls'
filenameDR = r'DeathRate.xls'
filenameGDP1 = r'GDP.xls'


def scrape(): #scrapes excel sheets and places into dataframes
    dfGDP = pd.read_excel(filenameGDP)
    dfGDP = dfGDP.fillna(-1)
    dfGDP.columns = dfGDP.columns.map(str)
    dfDR = pd.read_excel(filenameDR)
    dfDR = dfDR.fillna(-1)
    # return dfGDP, dfDR
    dfGDP1 = pd.read_excel(filenameGDP1)# for GDP not per Capita
    dfDR1 = pd.read_excel(filenameDR)# for DR with NaN
    return dfGDP, dfDR, dfGDP1, dfDR1


def getDates(dfGDP): #return list of dates for graphing purposes (1960-2020)
    time = []
    for (columnName, columnData) in dfGDP.iteritems():
        if columnName.isnumeric():
            time.append(columnName)
    return time


def getData(index, df): #return data of country at index in dataframe df
    data = []
    for (columnName, columnData) in df.iteritems():
        if columnName.isnumeric():
            data.append(columnData[index])
    return data


def getName(index, df): #return name of country at index in dataframe df
    for (columnName, columnData) in df.iteritems():
        if columnName == 'Country Name':
            name = columnData[index]
    return name


#seven regions
def getRegionGroups(region): #returns indexes of all tuples of given region - used with mappedlist (regionMap)
    df, dfDR, g, d = scrape()
    indexes = []
    for (columnName, columnData) in df.iteritems():
        if columnName == 'Region':
            for i in range(0, len(columnData)):
                if region == columnData[i]:
                    indexes.append(i)
    return indexes


def timeSeriesGDP(index, dfGDP): #get timeSeries for specific GDP graph
    time = getDates(dfGDP)
    dataGDP = getData(index, dfGDP)
    name = getName(index, dfGDP)
    plt.plot(time, dataGDP, label=name)
    title = name + ' GDP over Time'
    plt.title(title)
    plt.ylabel('GDP (current US Dollars)')
    plot_finalize()


def timeSeriesGDPregions(dfGDP): #get timeSeries for regions GDP graph
    time = getDates(dfGDP)
    regionMap = map(getRegionGroups, regions)
    for j in range(0, 7):
        currentIndexes = next(regionMap)
        for i in range(0, len(currentIndexes)):
            data = getData(currentIndexes[i], dfGDP)
            plt.plot(time, data, label=getName(currentIndexes[i], dfGDP))
        title = regions[j] + ' GDP over Time'
        plt.title(title)
        plt.ylabel('GDP (current US Dollars)')
        plot_finalize()
        

def timeSeriesDR(index, dfDR): #get timeSeries for specific DR graph
    time = getDates(dfDR)
    dataDR = getData(index, dfDR)
    name = getName(index, dfDR)
    plt.plot(time, dataDR, label=name)
    title = name + ' Death Rate over Time'
    plt.title(title)
    plt.ylabel('Crude Death Rate (per 1000 people)')
    plot_finalize()


def timeSeriesDRregions(dfDR): #get timeSeries for regions DR graph
    time = getDates(dfDR)
    regionMap = map(getRegionGroups, regions)
    for j in range(0, 7):
        currentIndexes = next(regionMap)
        for i in range(0, len(currentIndexes)):
            data = getData(currentIndexes[i], dfDR)
            plt.plot(time, data, label=getName(currentIndexes[i], dfDR))
        title = regions[j] + ' Death Rate over Time'
        plt.title(title)
        plt.ylabel('Crude Death Rate (per 1000 people)')
        plot_finalize()


def plot_finalize(): #finalize plot and show plot
    plt.margins(x=0)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 6})
    ax = plt.gca()
    ax.set_xticks(['1960', '1965', '1970', '1975', '1980', '1985', '1990', '1995', '2000', '2005', '2010', '2015', '2020'])
    plt.grid()
    plt.xlabel('Time')
    # plt.show()

# ####################################
def cleanData(dfGDP, dfDR):#get rid of the features besides IncomeGroup and GDP and Death rate
    
    DR_ = dfDR.drop(labels=['Country Code', 'IncomeGroup', 'Indicator Name','Indicator Code'], axis=1)
    DR_ = DR_.dropna(subset=["Region"])
    index = DR_.index
    DR_ = DR_.fillna(0)
    DR_ = DR_.drop(columns = ["Country Name"])
    
    
    GDP_ = dfGDP.drop(labels=['Country Code', 'Indicator Name','Indicator Code'], axis=1)
    
    for i in GDP_.index:
        if i not in index:
            GDP_ = GDP_.drop(index = i)
    
    GDP_ = GDP_.replace(-1, 0)
    GDP_ =  GDP_.fillna(0)
    GDP_ = GDP_.drop(columns = ["Country Name"])
    return GDP_, DR_
def getXy(GDP_, DR_):
    #normalizing the data and generated the X and y
    y_ = GDP_.loc[:, 'Region'].to_numpy()
    X1 = GDP_.iloc[:, 1:62]
    X1d = preprocessing.normalize(X1, axis=0)
    X2 = DR_.iloc[:, 1:62]
    X2d = preprocessing.normalize(X2, axis=0)
    X_ = np.concatenate((X1d, X2d), axis=1)
    return X_, y_
def getAvg(GDP_, DR_):
    avgGDP = GDP_.mean(axis=1)
    avgDR = DR_.mean(axis=1)
    Xavg_ = np.vstack((avgGDP, avgDR))
    return Xavg_
def MLPandcm(X_train_, X_test_, y_train_, y_test_):
    clf = MLPClassifier(activation='tanh', hidden_layer_sizes=(200,150), max_iter=1500)
    clf.fit(X_train_, y_train_)
    test_pred = clf.predict(X_test_)
    train_pred = clf.predict(X_train_)
    # print("MLP Classifier")
    # print("The testing accuracy score is: " + str(accuracy_score(y_test_, test_pred)))
    # print("The testing accuracy score for X_train is: " + str(accuracy_score(y_train_, train_pred_)))
    plot_confusion_matrix(clf, X_test_, y_test_)
    plt.xticks(rotation=45)
    # plt.show()
def DTandcm(X_train_, X_test_, y_train_, y_test_):
    clf = tree.DecisionTreeClassifier().fit(X_train_, y_train_)
    test_pred = clf.predict(X_test_)
    train_pred = clf.predict(X_train_)
    # print("Decision Tree Classifier")
    # print("The testing accuracy score is: " + str(accuracy_score(y_test_, clf.predict(X_test_))))
    # print("The testing accuracy score for X_train is: " + str(accuracy_score(y_train_, clf.predict(X_train_))))
    plot_confusion_matrix(clf, X_test_, y_test_)
    plt.xticks(rotation=45)
    # plt.show()
# ###########################################
def get_country_dictionary(gdp):
    countries = gdp.loc[:, 'Country Name']
    countriesDict = countries.to_dict()
    return countriesDict


def get_total(gdp, dr):
    GDP_ = gdp.loc[0, ~gdp.columns.isin(['Country Name', 'Country Code', 'Indicator Code', 'Region', 'IncomeGroup', 'Indicator Name'])]
    DR_ = dr.loc[0, ~dr.columns.isin(['Country Name', 'Country Code', 'Indicator Code', 'Region', 'IncomeGroup', 'Indicator Name'])]

    total_gdp_vals, total_dr_vals, total_country_index_vals = get_total_helper(GDP_, DR_, 0)
    # start i at 1
    for i in range(1, len(gdp)):
        index = i
        # trim columns to only get years/values
        GDP_ = gdp.loc[i, ~gdp.columns.isin(['Country Name', 'Country Code', 'Indicator Code', 'Region', 'IncomeGroup', 'Indicator Name'])]
        DR_ = dr.loc[i, ~dr.columns.isin(['Country Name', 'Country Code', 'Indicator Code', 'Region', 'IncomeGroup', 'Indicator Name'])]
        country_gdp, country_dr, country_index = get_total_helper(GDP_, DR_, index)

        # add return arrays to the end of our total storage arrays
        total_gdp_vals = np.concatenate((total_gdp_vals, country_gdp))
        total_dr_vals = np.concatenate((total_dr_vals, country_dr))
        total_country_index_vals = np.concatenate((total_country_index_vals, country_index))

    return total_gdp_vals, total_dr_vals, total_country_index_vals


def get_total_helper(gdp, dr, index):
    # merge gdp & dr about year
    data = pd.merge(gdp, dr, right_index=True, left_index=True)
    data.insert(2, 'index', index)
    data.columns = ['_gdp', '_dr', '_index']
    # trim rows where gdp = -1
    # data = data[data['_gdp'] != -1.0]
    data = data.fillna(-1)
    data = data.to_numpy()
    # split
    dataGDP, dataDR, country_index = np.hsplit(data, 3)

    return dataGDP, dataDR, country_index


def plot_regression(x_test, y_test_, y_pred, title):
    plt.scatter(x_test, y_test_)
    plt.plot(x_test, y_pred, color="black", linewidth=3)
    # add amount to y label(mil, bil, tril)
    plt.xlabel('Death Rate crude per 1000 people')
    plt.ylabel('GDP per capita (log)')
    plt.title(title)
    plt.show()
    plt.clf()


# line has terrible fit, perhaps separate gdp values based on gdp > x and gdp < x
def regression(gdp, dr, index):
    plt.clf()
    modified_data = np.concatenate((gdp, dr, index), axis=1)
    # remove -1 vals from both sets
    modified_data = np.delete(modified_data, np.where(modified_data[:, 0] == -1.0), axis=0)
    modified_data = np.delete(modified_data, np.where(modified_data[:, 0] < 0), axis=0)
    modified_data = np.delete(modified_data, np.where(modified_data[:, 1] == -1.0), axis=0)
    # removes outliers
    # modified_data = np.delete(modified_data, np.where(modified_data[:, 0] > 21040), axis=0)
    modified_data = np.delete(modified_data, np.where(modified_data[:, 1] > 20), axis=0)
    mod_gdp, mod_dr, mod_ind = np.hsplit(modified_data, 3)

    # apply log scale on gdp
    mod_gdp = np.log10(mod_gdp)

    # regression on all data points
    x_train, x_test, y_Train, y_Test = train_test_split(mod_dr, mod_gdp, test_size=0.2)
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_Train)
    y_pred = regr.predict(x_test)

    # compute the p-values
    import statsmodels.api as sm
    mod = sm.OLS(mod_dr, mod_gdp)
    fii = mod.fit()
    p_values = fii.summary2().tables[1]['P>|t|']
    print(p_values)

    # Evalute fit
    # 1 is perfect fit
    # print(r2_score(y_Test, y_pred))
    # print(regr.coef_)
    # print(mean_squared_error(y_Test, y_pred))

    plot_regression(x_test, y_Test, y_pred, 'Regression: All Countries')

    # REGRESSION ON UNITED STATES ##########################################
    modified_data = np.concatenate((gdp, dr, index), axis=1)
    # excel ind - 2 = country u want (remove -1/NaN from both sets if other country)
    modified_data = np.delete(modified_data, np.where(modified_data[:, 2] != 253-2), axis=0)
    mod_gdp, mod_dr, mod_ind = np.hsplit(modified_data, 3)
    # decide whether to keep or discard log scale for single country
    mod_gdp = np.log10(mod_gdp)

    x_train_, x_test_, y_Train_, y_Test_ = train_test_split(mod_dr, mod_gdp, test_size=0.50)
    regr = linear_model.LinearRegression()
    regr.fit(x_train_, y_Train_)
    y_pred_ = regr.predict(x_test_)

    # compute the p-values
    import statsmodels.api as sm
    mod = sm.OLS(y_pred_, x_test_)
    fii = mod.fit()
    p_values = fii.summary2().tables[1]['P>|t|']
    print(p_values)

    # Evalute fit
    # 1 is perfect fit
    # print(r2_score(y_Test, y_pred))
    # print(regr.coef_)
    # print(mean_squared_error(y_Test_, y_pred_))

    # plot_regression(x_test_, y_Test_, y_pred_, 'Regression: U.S.')


def single_year_regression(gdp, dr):
    # get 2020 data
    GDP_ = gdp.loc[:, gdp.columns.isin(['2015'])]
    DR_ = dr.loc[:, dr.columns.isin(['2015'])]

    data = pd.merge(GDP_, DR_, right_index=True, left_index=True).dropna().to_numpy()
    data = np.delete(data, np.where(data[:, 0] == -1.0), axis=0)
    data = np.delete(data, np.where(data[:, 1] == -1.0), axis=0)
    data = np.delete(data, np.where(data[:, 0] > 2E10), axis=0)

    GDP_, DR_ = np.hsplit(data, 2)

    GDP_ = np.log10(GDP_)

    # regression
    x_train, x_test, y_Train, y_Test = train_test_split(DR_, GDP_, test_size=0.2)
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_Train)
    y_pred = regr.predict(x_test)

    beta_hat = [regr.intercept_] + regr.coef_.tolist()

    # compute the p-values
    import statsmodels.api as sm
    mod = sm.OLS(y_pred, x_test)
    ft = mod.fit()
    p = ft.params
    p_values = ft.summary2().tables[1]['P>|t|']
    # print(p_values)


    # Evalute fit
    # 1 is perfect fit
    # print(r2_score(y_Test, y_pred))
    print(regr.coef_)
    print(mean_squared_error(y_Test, y_pred))

    plot_regression(x_test, y_Test, y_pred, 'Regression: All Countries Year 2015')


def kmeans_clustering(gdp, dr, index, diction):
    # format data into X[:,:] and y[:]
    plt.clf()
    data = np.concatenate((gdp, dr, index), axis=1)
    data = pd.DataFrame(data, columns=['GDP', 'DR', 'Index']).dropna().to_numpy()
    data = np.delete(data, np.where(data[:, 0] == -1.0), axis=0)
    data = np.delete(data, np.where(data[:, 1] == -1.0), axis=0)
    x1, x2, y_ = np.hsplit(data, 3)
    x1 = np.log10(x1)
    X_ = np.concatenate((x1, x2), axis=1)

    # kmeans
    kmeans = KMeans(init='random', n_clusters=10)
    kmeans.fit(X_)

    kmeans.labels_ = y_
    y_kmeans = kmeans.predict(X_)

    # create dictionary for new y_kmeans to track country's new index
    length = len(diction)
    ldfds = list(['null'])
    d = dict()
    d[0] = ldfds

    # print(diction[1])
    for i in range(0, length):
        list_entry = list([diction[i]])
        if y_kmeans[i] not in d:
            d[y_kmeans[i]] = list_entry
        else:
            d[y_kmeans[i]].append(diction[i])

    del d[0][0]
    # print(d)
    colors = np.array(['blue', 'orange', 'purple', 'brown', 'red', 'yellow', 'blueviolet', 'black', 'green', 'grey'])

    # plot
    scatter = plt.scatter(X_[:, 1], X_[:, 0], c=colors[y_kmeans])

    plt.xlabel('Death Rate crude per 1000 people')
    plt.ylabel('GDP per capita (log)')
    plt.title('K-means Clustering')
    plt.show()


if __name__ == '__main__':
    df_GDP, df_DR, df_GDP1, df_DR1 = scrape()
    timeSeriesDR(54, df_DR)
    timeSeriesGDP(76, df_GDP)
    # timeSeriesGDPregions(df_GDP)
    # timeSeriesDRregions(df_DR)

    # ############################################################
    total_gdp, total_dr, total_country_index = get_total(df_GDP, df_DR)

    regression(total_gdp, total_dr, total_country_index)
    single_year_regression(df_GDP, df_DR)
    dictionary = get_country_dictionary(df_GDP)
    kmeans_clustering(total_gdp, total_dr, total_country_index, dictionary)

    # #############################################################
    GDP, DR = cleanData(df_GDP1, df_DR1)
    X, y = getXy(GDP, DR)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    MLPandcm(X_train, X_test, y_train, y_test)
    DTandcm(X_train, X_test, y_train, y_test)

    Xavg = getAvg(GDP, DR)
    X_avg_train, X_avg_test, y_avg_train, y_avg_test = train_test_split(Xavg.transpose(), y, test_size=0.2)
    MLPandcm(X_avg_train, X_avg_test, y_avg_train, y_avg_test)
    DTandcm(X_avg_train, X_avg_test, y_avg_train, y_avg_test)
