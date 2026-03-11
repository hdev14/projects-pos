import seaborn as sns
import matplotlib.pyplot as plt

data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]

plt.figure(figsize=(10, 6))
sns.histplot(data)
plt.title('Histogram of Data')
plt.show()