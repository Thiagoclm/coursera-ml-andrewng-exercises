import matplotlib.pyplot as plt
import numpy as np

#Plotting a sigmoid (logistic) function

x = np.linspace(-10,10,10000)
g_x = 1/(1+np.e**-x)


fig, ax = plt.subplots(1,1,figsize=(5,3))

ax.plot(x,g_x, c='b')
ax.set_title('Sigmoid (logistic) function')