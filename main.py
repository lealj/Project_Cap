# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

regions = ['Latin America & Caribbean', 'South Asia', 'Sub-Saharan Africa', 'Europe & Central Asia', 'Middle East & North Africa', 'East Asia & Pacific', 'North America']
filenameGDP = r'GDP.xls'
filenameDR = r'DeathRate.xls'

def scrape(): #scrapes excel sheets and places into dataframes
    dfGDP = pd.read_excel(filenameGDP)
    dfDR = pd.read_excel(filenameDR)
    return dfGDP, dfDR

def getDates(dfGDP): #return list of dates for graphing purposes (1960-2020)
    time = []
    for (columnName, columnData) in dfGDP.iteritems():
        if (columnName.isnumeric()):
            time.append(columnName)
    return time

def getData(index, df): #return data of country at index in dataframe df
    data = []
    for (columnName, columnData) in df.iteritems():
        if (columnName.isnumeric()):
            data.append(columnData[index])
    return data

def getName(index, df): #return name of country at index in dataframe df
    for (columnName, columnData) in df.iteritems():
        if (columnName == 'Country Name'):
            name = columnData[index]
    return name

#seven regions
def getRegionGroups(region): #returns indexes of all tuples of given region - used with mappedlist (regionMap)
    df, dfDR = scrape()
    indexes = []
    for (columnName, columnData) in df.iteritems():
        if (columnName == 'Region'):
            for i in range(0, len(columnData)):
                if region == columnData[i]:
                    indexes.append(i)
    return indexes

def timeSeriesGDP(index): #get timeSeries for specific GDP graph
    dfGDP, dfDR = scrape()
    time = getDates(dfGDP)
    dataGDP = getData(index, dfGDP)
    name = getName(index, dfGDP)
    plt.plot(time, dataGDP, label=name)
    title = name + ' GDP over Time'
    plt.title(title)
    plt.ylabel('GDP (current US Dollars)')
    plot_finalize()

def timeSeriesGDPregions(): #get timeSeries for regions GDP graph
    dfGDP, df = scrape()
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
        

def timeSeriesDR(index): #get timeSeries for specific DR graph
    dfGDP, dfDR = scrape()
    time = getDates(dfDR)
    dataDR = getData(index, dfDR)
    name = getName(index, dfDR)
    plt.plot(time, dataDR, label=name)
    title = name + ' Death Rate over Time'
    plt.title(title)
    plt.ylabel('Crude Death Rate (per 1000 people)')
    plot_finalize()


def timeSeriesDRregions(): #get timeSeries for regions DR graph
    df, dfDR = scrape()
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
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", prop={'size': 6})
    ax = plt.gca()
    ax.set_xticks(['1960', '1965', '1970', '1975', '1980', '1985', '1990', '1995', '2000', '2005', '2010', '2015', '2020'])
    plt.grid()
    plt.xlabel('Time')
    plt.show()


def get_total(gdp, dr):
    # intialize arrays to store entirety of gdp & dr values
    GDP = gdp.loc[0, ~gdp.columns.isin(['Country Name', 'Country Code', 'Indicator Code', 'Region', 'IncomeGroup', 'Indicator Name'])]
    DR = dr.loc[0, ~dr.columns.isin(['Country Name', 'Country Code', 'Indicator Code', 'Region', 'IncomeGroup', 'Indicator Name'])]

    total_gdp_vals, total_dr_vals = get_total_helper(GDP, DR)
    # start i at 1
    for i in range(1, len(gdp)):
        # trim columns to only get years/values
        GDP = gdp.loc[i, ~gdp.columns.isin(['Country Name', 'Country Code', 'Indicator Code', 'Region', 'IncomeGroup', 'Indicator Name'])]
        DR = dr.loc[i, ~dr.columns.isin(['Country Name', 'Country Code', 'Indicator Code', 'Region', 'IncomeGroup', 'Indicator Name'])]
        country_gdp, country_dr = get_total_helper(GDP, DR)

        # add return arrays to the end of our total storage arrays
        total_gdp_vals = np.concatenate((total_gdp_vals, country_gdp))
        total_dr_vals = np.concatenate((total_dr_vals, country_dr))

    return total_gdp_vals, total_dr_vals


def get_total_helper(gdp, dr):
    # merge gdp & dr about year
    data = pd.merge(gdp, dr, right_index=True, left_index=True)
    data.columns = ['_gdp', '_dr']

    # trim rows where gdp = -1
    data = data[data['_gdp'] != -1.0]
    data = data.to_numpy()

    # split
    dataGDP, dataDR = np.hsplit(data, 2)

    return dataGDP, dataDR


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_GDP, df_DR = scrape()
    timeSeriesDR(54)
    timeSeriesGDP(76)
    timeSeriesGDPregions()
    timeSeriesDRregions()

    total_gdp, total_dr = get_total(df_GDP, df_DR)


