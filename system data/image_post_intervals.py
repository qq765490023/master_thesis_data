import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy import stats
import seaborn as sns


# 读取文件，假设文件名为 'GPT_time.txt'
data_list = []
with open('image_post_intervals.txt', 'r') as file:
    for line in file:
        data_list.append(float(line.strip()))

# 转换为DataFrame
data = pd.DataFrame(data_list, columns=['time'])

# 计算Z分数
data['z_time'] = zscore(data['time'])

# 原始样本数
original_samples = len(data)

# 剔除离群值（例如，Z分数的绝对值大于2）
filtered_data = data[(data['z_time'].abs() <= 2)]

# 剔除离群值后的样本数
filtered_samples = len(filtered_data)

k2, p = stats.normaltest(data_list)
alpha = 0.05
if p < alpha:
    print("数据不符合正态分布")
else:
    print("数据可能符合正态分布")
sns.distplot(data_list, hist=True, kde=True)
plt.show()

# 处理时间的平均值
avg_time = filtered_data['time'].mean()
median_time = filtered_data['time'].median()
print('Original Samples:', original_samples)
print('Filtered Samples:', filtered_samples)
print('Average Time:', avg_time)
print('Median Time:', median_time)

# 画散点图
plt.scatter(filtered_data.index, filtered_data['time'])
plt.xlabel('Index')
plt.ylabel('Time (ms)')
plt.title('Time interval between receiving images')
plt.show()
