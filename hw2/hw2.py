
import json
import urllib
import os
import time

dir_out = '/home/GA14/data/yahoo_finance_data'

if not os.path.exists(dir_out):
    os.mkdir(dir_out)
    print 'created directory %s' % dir_out

threshold = 25
symbols = ['FB']
tmp =[]

for symbol in symbols:
    time.sleep(.5)
    data_out = {}
    threshold_l = 'less than %s' % threshold
    threshold_h = 'more than %s' % threshold
    try:
        if threshold_l not in data_out:
            data_out[threshold_l]=0
        if threshold_h not in data_out:
            data_out[threshold_h]=0
        url = 'http://ichart.finance.yahoo.com/table.csv?s=%s&d=9&e=24&f=2013&g=d&a=9&b=18&c=2012&ignore=.csv' % (symbol)
        data = urllib.urlopen(url).read()
        print data
        file = '%s/%s_data.csv' % (dir_out, symbol)
        print 'savingfile %s' % file
        f = open(file,'w')
        f.write('%s' % str(data))
        f.close()
        data = data.split('\n')
        for d in data[1:-1]:
            try:
                d = d.split(',')
                close = d[6]
                tmp.append(close)
                if float(close)>=float(threshold):
                    data_out[threshold_h]+=1
                elif float(close)<float(threshold):
                    data_out[threshold_l]+=1
            except:pass
    except:pass
#    print json.dumps(data_out)

import pylab as pl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
symbol = 'FB'
fb_file = '%s/%s_data.csv' % (dir_out, symbol)

read_file = pd.read_csv(fb_file)
file_len = len(read_file)

#leave out close because it's correlated
data = read_file.drop(['Date', 'Close', 'Adj Close'], axis=1)
y = read_file['Adj Close']

X_train = data[:-file_len*0.3]
y_train = y[:-file_len*0.3]

X_test = data[-file_len*0.3:]
y_test = y[-file_len*0.3:]

#create linear regression object
model = LinearRegression()

#train the model using the training data set
model.fit(X_train, y_train)

#coefficients and intercept
print '\n The coefficients are: \n', model.coef_
print '\n The intercept is: \n', model.intercept_

#mean square error
print ('\n Residual sum of squares: %.2f \n' % np.mean((model.predict(X_test) - y_test) **2))

#score(X y) returns the R^2 of the prediction, or 1-u/v, u = ((y_true-y_predicted)**2).sum(), v = ((y_true-y_true.mean())**2).sum()
#best score is 1.0, lower scores are worse
print ('\n R^2, coefficient of determination is: %.2f \n' % model.score(X_test, y_test))

<<<<<<< HEAD
=======

>>>>>>> temp_branch
coeff =model.coef_

sample_data = read_file[read_file['Date'] == '2013-09-24']
pred_y = coeff[0]*sample_data['Open'] + coeff[1]*sample_data['High'] + coeff[2]*sample_data['Low'] + coeff[3]*sample_data['Volume'] + model.intercept_

sample_y = sample_data['Adj Close']

print ('\n For 2013-09-24, the actual Adj Close is: %.2f' % sample_y + ' and the predicted Adj Close is: %.2f' %  pred_y)
