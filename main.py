#######################################
#Lino Valdovinos
#UCR Cs-171 Machine Learning and Data Science
#Assignment 1

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Begin of Question(0)
#read the wine data and place in a numpy array and list of classes
def wine_loader(f):
    file = open(f)
    instances = np.array([])
    in_class = []
    for line in file:
        features = line.split(',')
        classify = features[0:1]
        features = features[1:]
        features = [float(i) for i in features]
        instances = np.append(instances, features)
        in_class.append(classify)
    return instances, in_class

# read the iris data and place in in a numpy array and a list of names
def iris_loader(f):
    file = open(f)
    instances = np.array([])
    names = []
    for line in file:
        features = line.split(',',)
        name = features[4:]
        features = features[:-1]
        features = [float(i) for i in features]
        instances = np.append(instances,features)
        names.append(name)
    return instances, names
#END OF Question(0)



#Begin of Question(1)
#fill up the bins with respects to the data to the data
def fillbins(bins, data):
    bin_count = [0]*len(bins)
    for i in range(len(data)):
        for j in range(len(bins)-1):
            if(data[i] >= bins[j] and data[i] < bins[j+1]):
                print(str(i)+ " " + str(data[i]) + ' goes into '+ str(bins[j]) +"-"+str(bins[j+1]))
                bin_count[j] += 1
            elif(j == (len(bins)-2) and data[i] >= bins[j+1]):
                bin_count[j+1] +=1
    return bin_count

#implementation of histograms
def histogram(numbins, data, feature_name ):
    hismax = np.amax(data)
    hismin = np.amin(data)
    bin_width = (hismax - hismin)/float(numbins)
    bins = [hismin]
    r1 = hismin
    r2 = hismin + bin_width
    X = []
    for i in range(1,numbins+1):
        string = str(round(r1,3))+"-"+str(round(r2,3))
        X.append(string)
        bins.append(round(bins[i-1] + bin_width,3))
        r1 = r2
        r2 = r2 + bin_width

    # print(X)
    # print(data)
    inds = np.arange(len(bins))
    bin1_height= fillbins(bins, data[0:59])
    bin2_height= fillbins(bins, data[59:130])
    bin3_height= fillbins(bins, data[130:178])

    f = plt.figure(1)
    setosa = plt.bar(inds,bin1_height, .3, color = 'darkred', edgecolor = 'black', label = 'class 1')
    versicolour = plt.bar(inds-.3,bin2_height, .3, color = 'ivory', edgecolor = 'black', label = 'class 2')
    virginica = plt.bar(inds+.3,bin3_height, .3, color = 'lightpink', edgecolor = 'black', label = 'class 3')
    #plt.hist( data, bins,width = .1, color = 'lightcoral', edgecolor = 'black')
    plt.xlabel(feature_name)
    plt.ylabel('Amount in set')
    plt.title('Distribution for '+ feature_name + ' in Wine Dataset.')
    plt.xticks(inds, X, rotation = 'vertical')
    plt.legend(handles=[setosa, versicolour, virginica])
    f.show()
#End of Question(1)

#End of Question(2.1)
def mean(x):
    m = 0;
    for i in range(len(x)):
        m += x[i]
    return(m/len(x))

def correlation(x,y):

    x_bar = mean(x)
    y_bar = mean(y)
    #print("x_bar "+ str(x_bar) + " y_bar " + str(y_bar))

    num = 0.0
    den1 = 0.0
    den2 = 0.0

    for i in range(len(x)):
        num += (x[i]-x_bar)*(y[i]-y_bar)
        den1 += ((x[i]-x_bar)**2)
        den2 += ((y[i]-y_bar)**2)

    return( num /((den1*den2) **.5))

def fill_correlation_matrix(data):
    corr = np.array([])
    for i in range(np.size(data,1)):
        for j in range(np.size(data,1) ):
            if(i == j):
                corr = np.append(corr, 1 )
            else:
                n = correlation(data[:,i],data[:,j])
                print("correlation betweeen " +str(i) + " and " +str(j) + " is " + str(n))
                corr = np.append(corr,n)
    corr = corr.reshape(13,13)
    print(corr)

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    plt.title("Correlation in Wine features ")
    with sns.axes_style("white"):
        ax = sns.heatmap(corr, mask = mask, vmax = 1, square = True, cbar_kws = {"shrink":.5}, cmap="YlGnBu")
        plt.show()

#end of Question(2.1)


#Begin of Question(2.2)
def scatterplots(data, data_class):
    c = np.full((50,1),50)
    c = np.append(c, np.full((50,1),100) )
    c = np.append(c, np.full((50,1),150) )

    x = data[:,2]
    y = data[:,3]
    plt.xlabel('petal lengthh')
    plt.ylabel('petal width')
    plt.title('purple = Setosa, green = Versicolour, yellow = Virginica' )
    #plt.legend(handles=[x,y])
    plt.scatter(x,y, s = 40, c = c)
    plt.show()

#End of  Question(2.2)

#Begin of  Question(2.3)
def distance(x,y,p):
    distance = 0;
    for i in range(len(x)):
        distance = distance + (x[i] -y[i])**2
    if(p==1):
        return(distance)
    if(p == 2):
        return((distance)**(.5))


def fill_distance_map(data, p):
    s = len(data)
    heatmap = np.zeros((s,s))
    print(heatmap)
    for i in range(s):
        for j in range(s):
            heatmap[i][j] = distance(data[i],data[j],p)
    print(heatmap)


    plt.title("Distance heatmap for the Wine Dataset unsing p = " + str(p))
    with sns.axes_style("white"):
        ax = sns.heatmap(heatmap, square = True, cbar_kws = {"shrink":.5})
        plt.show()

#End of  Question(2.3)

#Begin of  Question(2.4)
def nearest_class(data, data_class, p):
    s = len(data)
    labels = np.zeros((s))
    dist = 1000000.0
    point = 0.0
    count = 0.0
    for j in range(s):
        for k in range(s):
            if(j!=k):
                temp = distance(data[j],data[k],p)
                if ( temp < dist):
                    dist = temp
                    point = k
        if data_class[point] == data_class[j]:
            labels[j] = 1
            count +=1
        else:
            labels[j] = 0

    print(count/s)
    print(labels)

#End of  Question(2.4)


if __name__ == '__main__':


    iris_data, iris_class = iris_loader('iris_data.txt')
    wine_data, wine_class = wine_loader('wine_data.txt')
    iris_data = iris_data.reshape(150, 4)
    wine_data = wine_data.reshape(178, 13)

    nearest_class(iris_data, iris_class, 2)
    #fill_distance_map(wine_data,2)

    #scatterplots(iris_data, iris_class)

    #print(correlation(iris_data[:,0],iris_data[:,0]))
    #fill_correlation_matrix(wine_data)
    #print(iris_data)
    #print(iris_data[0:50,0])

    #histogram(100,wine_data[0:178,2], 'Ash' )

    # g = plt.figure(2)
    # plt.ylabel('Ash')
    # plt.title('Box-plot for Ash in wine Dataset.')
    # sepal_data = [iris_data[0:59,2],iris_data[59:130,2],iris_data[130:178,2]]
    # plt.boxplot(sepal_data, labels = ('class 1','class 2','class 3'))
    # g.show()
    #

    raw_input()
