import os
import numpy as np
import matplotlib.pyplot as plt

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




if __name__ == '__main__':


    iris_data, iris_class = iris_loader('iris_data.txt')
    wine_data, wine_class = wine_loader('wine_data.txt')
    iris_data = iris_data.reshape(150, 4)
    wine_data = wine_data.reshape(178, 13)

    #print(iris_data)
    #print(iris_data[0:50,0])

    histogram(100,wine_data[0:178,2], 'Ash' )

    # g = plt.figure(2)
    # plt.ylabel('Ash')
    # plt.title('Box-plot for Ash in wine Dataset.')
    # sepal_data = [iris_data[0:59,2],iris_data[59:130,2],iris_data[130:178,2]]
    # plt.boxplot(sepal_data, labels = ('class 1','class 2','class 3'))
    # g.show()
    #
    raw_input()


    #plt.plot(iris_data[1])
    #plt.show()
