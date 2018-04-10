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

            if(data[i] > bins[j] and data[i] <= bins[j+1]):
                print(str(i)+ " " + str(data[i]) + ' goes into '+ str(bins[j]))
                bin_count[j] += 1
            elif(j == (len(bins)-2) and data[i] > bins[j+1]):
                print(str(i)+ " " + str(data[i]) + ' goes into '+ str(bins[j+1]))
                bin_count[j+1] += 1
    return bin_count

#implementation of histograms
def histogram(numbins, data, feature_name, class_name ):
    hismax = np.amax(data)
    hismin = np.amin(data)
    bin_width = float((hismax - hismin)/numbins)
    bins = [hismin]
    r1 = hismin
    r2 = hismin + bin_width
    string = str(round(r1,3))+"-"+str(round(r2,3))
    X = []
    X.append(string)

    for i in range(1,numbins):
        bins.append(bins[i-1] + bin_width)
        r1 = r2
        r2 = r2 + bin_width
        string = str(round(r1,3))+"-"+str(round(r2,3))
        X.append(string)

    print(X)
    print(data)
    inds = np.arange(len(bins))
    bin_height= fillbins(bins, data)
    plt.bar(inds,bin_height,bin_width + .5, color = 'lightcoral', edgecolor = 'black')
    plt.xlabel('Feature Distribution')
    plt.ylabel(feature_name)
    plt.title('Distribution for '+ feature_name + ' in '+ class_name)
    plt.xticks(inds, X )
    plt.show()


if __name__ == '__main__':


    iris_data, iris_class = iris_loader('iris_data.txt')
    wine_data, wine_class = wine_loader('wine_data.txt')
    iris_data = iris_data.reshape(150, 4)
    wine_data = wine_data.reshape(178, 13)

    #print(iris_data)
    #print(iris_data[0:50,0])

    histogram(5,iris_data[0:50,0], 'sepal length', 'Iris Setosas' )



    #plt.plot(iris_data[1])
    #plt.show()
