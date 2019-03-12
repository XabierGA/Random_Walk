import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
from mpl_toolkits.mplot3d import Axes3D


N=10000

X = np.zeros(N)
Y = np.zeros(N)
Z = np.zeros(N)

X[0] = 50
Y[0] = 50
Z[0] = 50

def random_walk(N , x , y ,z, radius):
    for i in range(1, N):
        rand = np.random.uniform(-1, 1, 1)
        T = np.sign(rand)
        #print(C)
        if np.abs(rand)>0.66:
            x[i] = x[i-1] + T
            y[i] = y[i-1]
            z[i] = z[i-1]
        elif (np.abs(rand)>0.33 and np.abs(rand)<0.66):
            y[i] = y[i-1] + T
            x[i] = x[i-1]
            z[i] = z[i-1]
        else:
            y[i] = y[i-1]
            x[i] = x[i-1]
            z[i] = z[i-1] + T

    return x,y,z




fig = plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
ax = Axes3D(fig)
plt.title("3D Random  Walks")
for i in range(10):
    X = np.zeros(N)
    Y = np.zeros(N)
    Z = np.zeros(N)
    X[0] = 50
    Y[0] = 50
    Z[0] = 50
    X , Y , Z = random_walk(N , X , Y , Z , 300)
    plt.plot(X, Y, Z,alpha=0.6 )
    ax.scatter(X[-1],Y[-1],Z[-1])
plt.savefig("3d.pdf")
plt.show()



# ax = plt.subplot(1,1,1, projection='3d')
# cm = plt.get_cmap('jet')
# ax.set_prop_cycle('color',[cm(1.*i/(x.shape[-1]-1)) for i in range(x.shape[-1]-1)])
# for i in range(x.shape[-1]-1):
#     ax.plot([x[i+1],x[i]], [y[i+1],y[i]], [z[i+1],z[i]],alpha=0.6)
# ax.scatter(x[-1],y[-1],z[-1],facecolor=cm(1))
# plt.savefig('3d_random_walk_static.png', bbox_inches='tight', pad_inches=0.02, dpi=250)
# plt.show()
