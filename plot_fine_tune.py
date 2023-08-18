import pandas as pd
import matplotlib.pyplot as plt

dat = pd.read_csv('ffhq_ft.csv')

print(dat)

plt.plot(dat['Step'],dat['Value'])
plt.xlabel('Steps', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.savefig('FFHQ_fine_tune.png')
plt.show()