import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
from scipy.optimize import curve_fit

N = 1000000
number_walks = 10
X = np.zeros(N)
Y = np.zeros(N)
X[0] = 50
Y[0] = 50

def function(x, a , b):
    return a*np.sqrt(x)+b


def random_walk(N , x , y , radius):
    for i in range(1, N):
        rand = np.random.uniform(-1, 1, 1)
        T = np.sign(rand)
        #print(C)
        if np.abs(rand)>0.5:
            x[i] = x[i-1] + T
            y[i] = y[i-1]
        else:
            y[i] = y[i-1] + T
            x[i] = x[i-1]
        distance = np.sqrt(x[i]**2+y[i]**2)

        # while distance > radius:
        #     rand = np.random.uniform(-1, 1, 1)
        #     T = np.sign(rand)
        #     if rand>0.5:
        #         x[i] = x[i-1] + T
        #         y[i] = y[i-1]
        #     else:
        #         y[i] = y[i-1] + T
        #         x[i] = y[i-1]
        #     distance = np.sqrt(x[i]**2+y[i]**2)
    return x,y


#fig = plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
#plt.title("2-D Random Walks")
#fig.text(0.8, 0.8, 'N = 1000', color='blue',
#        bbox=dict(facecolor='none', edgecolor='blue', pad=10.0))
#for i in range(number_walks):
#    result = random_walk(N , X , Y , 100)
#    plt.plot(result[0] , result[1])
#result = random_walk(N , X , Y , 1000)
#plt.plot(result[0] , result[1] , label = 'N = %i' %N)
#plt.legend(loc=2)
# distance = []
#for i in range(10000):
#     X , Y = random_walk(1000 , X , Y ,10)
#     distance.append(np.sqrt((X[-1]-X[0])**2 + (Y[-1] - Y[0])**2 ))
# print("Length" , np.sqrt(N))
# print(np.mean(distance))
# print(np.std(distance))
# print(len(distance))
#plt.savefig("random_walk_1000000.pdf")
#plt.show(True)

N_mean = np.arange(10, 1000 , 10)
mean = []
std = []
for number in N_mean:
    distance = []

    for i in range(1000):
        X = np.zeros(int(number))
        Y = np.zeros(int(number))
        X[0] = 50
        Y[0]= 50
        X, Y = random_walk(int(number) , X , Y , 1000)
        distance.append((X[-1]-X[0])**2 + (Y[-1] - Y[0])**2 )
        #print(distance)
    #print(np.mean(distance))
    mean.append(np.sqrt(np.mean(distance)))
    std.append(np.std(distance))
for i in range(len(std)):
    std[i] = np.sqrt(np.log(mean[i]))/2.0


param, popt = curve_fit(function, N_mean , mean)
points = np.linspace(min(N_mean) , max(N_mean) , 10000)
y_fit = function(points , *param)

fig = plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(N_mean , mean , xerr = 0, yerr = std , fmt = 'go' , ecolor = 'r' , markersize = 6 , label = "Measured times")
plt.plot(points , y_fit , 'b-', label = "Square root fit")
#plt.plot(N_mean , mean ,'mo')
plt.title("Verifying proportionality of RMS for 2D Random Walks")
plt.xlabel("N")
plt.legend(loc = 2)
plt.ylabel("$\sqrt{<R^{2}(N)>}$")
plt.savefig("proof_2dimensional.pdf")
plt.show(True)
