import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, zscore
import statsmodels.api as sm
import json
import numpy as np

data_list = []
with open('GPT_time.txt', 'r') as file:
    for line in file:
        data_list.append(json.loads(line))

data = pd.DataFrame(data_list)

# 计算Z分数
data['z_time'] = zscore(data['time'])
data['z_length'] = zscore(data['length'])

# 原始样本数
original_samples = len(data)

# 剔除离群值（例如，Z分数的绝对值大于2）
filtered_data = data[(data['z_time'].abs() <= 2) & (data['z_length'].abs() <= 2)]

# 剔除离群值后的样本数
filtered_samples = len(filtered_data)

# 处理时间和图像大小的平均值
avg_time = filtered_data['time'].mean()
avg_length = filtered_data['length'].mean()

print('Original Samples:', original_samples)
print('Filtered Samples:', filtered_samples)
print('Average Time:', avg_time)
median_time = filtered_data['time'].median()
print('Median Time:', median_time)
print('Average Length:', avg_length)



correlation, _ = pearsonr(filtered_data['length'], filtered_data['time'])
print('Pearson correlation:', correlation)

x = filtered_data['length']
y = filtered_data['time']
if abs(correlation) > 0.6:
    # 2. 找出线性关系的数值
    slope, intercept = np.polyfit(x, y, 1)
    print('Slope:', slope, 'Intercept:', intercept)

    # 3. 画图
    plt.scatter(x, y, label='Data Points')
    plt.plot(x, slope * x + intercept, label='Fitted Line', color='red')
    plt.legend()
    plt.xlabel('Number of words (n)')
    plt.ylabel('Time (ms)')
    plt.title('GPT generating time')
    plt.show()
else :
    # 散点图
    sns.scatterplot(x='length', y='time', data=filtered_data)
    plt.show()

    # 相关分析
    print("No strong linear relationship found.")



