import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import svm

dates = []
prices = []


def get_data(HistoricalQuotes):
    with open('HistoricalQuotes.csv','r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[3]))
    return





def predict_price(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

     # defining the support vector regression models
     svr_lin = svm.SVR(kernel= 'linear', C= 1e3)

     # fitting the data points in the models
     svr_lin.fit(dates, prices)


    # plotting the initial datapoints

    plt.scatter(dates, prices, color= 'black', label= 'Data')

    # plotting the line made by linear kernel

    plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_lin.predict(x)[0]


get_data('HistoricalQuotes.csv')


predicted_price = predict_price(dates, prices, 29)


print(predicted_price)
