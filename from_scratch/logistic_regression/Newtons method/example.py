from matplotlib import pyplot as plt
import numpy as np

values = []

X = np.linspace(0.0,1.0,num=50)
Y = -np.log(X)

figure = plt.figure()

axe_1 = figure.add_subplot(111)
axe_1.plot(X,Y,label="- Log(X)")
axe_1.set_xbound(0,1)
axe_1.set_ylabel("- Log(h(X))")
axe_1.set_xlabel('h(X)')
axe_1.set_title('When Y=1 ')

plt.show()