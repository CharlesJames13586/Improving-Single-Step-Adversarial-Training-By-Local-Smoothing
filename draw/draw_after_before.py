import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname="result/FGSM.csv", delimiter=',')
iters = list(range(13))[1:]

plt.plot(iters, data[4], color=(0, 0.3125, 0.937), marker='o', label="the robust model")
bottom = data[4][:] - data[5][:]
up = data[4][:] + data[5][:]
plt.fill_between(iters, bottom, up, color=(0, 0.3125, 0.937, 0.1))

plt.plot(iters, data[9], color=(0, (8*16+10)/255, 0), marker='o', label="the catastrophic ovrfitting model")
bottom = data[9][:] - data[10][:]
up = data[9][:] + data[10][:]
plt.fill_between(iters, bottom, up, color=(0, (8*16+10)/255, 0, 0.1))
plt.legend(fontsize=12)
plt.xlabel('$\ell_{\infty}\\alpha_{test}$', fontsize=12)
plt.ylabel('accuracy(%)', fontsize=12)
plt.savefig("test1.png", dpi=400)
plt.savefig("test1.pdf", format="pdf")
