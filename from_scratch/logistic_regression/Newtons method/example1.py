from matplotlib import pyplot as plt
import numpy as np

values = []

X = np.linspace(0.0,1.0,num=50)
Y = -np.log(1-X)

figure = plt.figure()

axe_2 = figure.add_subplot(111)
axe_2.plot(X,Y,label="- Log(1-X)")
axe_2.set_xbound(0,1)
axe_2.set_xlabel('h(X)')
axe_2.set_ylabel("- Log(1-h(X))")
axe_2.set_title('When Y=0 ')

plt.show()