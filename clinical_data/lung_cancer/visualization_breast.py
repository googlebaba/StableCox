import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from random import sample
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
#training_pd_data = pd.read_csv('../nature_survival.csv', index_col=0)
#test0_pd_data = pd.read_csv('../cell_survival.csv', index_col=0)
#test1_pd_data = pd.read_csv('../NC_survival.csv', index_col=0)
#training_pd_data = pd.read_csv('../breast_cancer/breast_train_survival.csv', index_col=0)
#test0_pd_data = pd.read_csv('../breast_cancer/breast_test1_survival.csv', index_col=0)
#test1_pd_data = pd.read_csv('../breast_cancer/breast_test2_survival.csv', index_col=0)

#training_data = pd.read_csv('./train_pd.csv', index_col=0)
test0_data = pd.read_csv('./test4_pd_visual.csv', index_col=0)
test1_data = pd.read_csv('./test5_pd_visual.csv', index_col=0)
#test3_data = pd.read_csv('../medical/test3.csv', index_col=0)
    
#test4_data = pd.read_csv('../medical/test4.csv', index_col=0)
#test5_data = pd.read_csv('../medical/test5.csv', index_col=0)
#test6_data = pd.read_csv('../medical/test6.csv', index_col=0)
#test7_data = pd.read_csv('../medical/test7.csv', index_col=0)

#training_pd_data = training_pd_data.drop(['Survival.months', 'Survival.status', 'Cohort'], axis=1)

#test0_pd_data = test0_pd_data.drop(['Survival.months', 'Survival.status', 'Cohort'], axis=1)
#test1_pd_data = test1_pd_data.drop(['Survival.months', 'Survival.status', 'Cohort'], axis=1)


#training_data = training_data.drop(['2', "Reccur.status", "Reccur.months"], axis=1)
#test0_data = test0_data.drop(["Survival.status", "Survival.months"], axis=1)
#test1_data = test1_data.drop(["Survival.status", "Survival.months"], axis=1)
#test0_data.to_csv("./central.csv", index=False)
#test1_data.to_csv("./peripheral.csv", index=False)
#train_np = np.array(training_data[training_data.columns[:-2]])

def confidence_ellipse(x, y, ax, n_std=2.4477, facecolor='none', **kwargs):
    """
    创建一个表示给定数据的n标准差协方差椭圆的路径补丁。
    n_std: 对应于95%置信区间的标准差数。对于2D数据，2.4477标准差覆盖约95%的点。
    """
    if x.size != y.size:
        raise ValueError("x和y必须大小相同")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])  # 相关系数
    # 使用特征值计算椭圆的旋转角度和半轴长度
    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)
    ellipse_radius_x = np.sqrt(1 + pearson) * n_std
    ellipse_radius_y = np.sqrt(1 - pearson) * n_std
    ellipse = Ellipse((0, 0), width=ellipse_radius_x * 2, height=ellipse_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # 计算椭圆的旋转角度
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    transf = transforms.Affine2D()\
        .rotate_deg(45)\
        .scale(scale_x, scale_y)\
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

test0_np = np.array(test0_data[test0_data.columns[:-4]])
test1_np = np.array(test1_data[test1_data.columns[:-4]])


pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=3)
all_data = np.concatenate((test0_np, test1_np))
all_data = np.nan_to_num(all_data)
all_data = tsne.fit_transform(all_data)

test1 = all_data[:test0_np.shape[0]]
test2 = all_data[test0_np.shape[0]:test0_np.shape[0]+test1_np.shape[0]]


print(test1.shape)
print(test2.shape)

camp2 = sns.color_palette("Set2")
colors1 = sample(camp2, 4)
fig, ax = plt.subplots()

colors1 = ['#F39B7FCC', '#91D1c2cC', '#ADB6B6B2', "#925E9FB2"]
colors2 = ['#F39B7F50', '#91D1c250', '#ADB6B650', "#925E9F50"]

#plt.scatter(train[:, 0], train[:, 1], color=colors1[0], label=f'Cohort1')
ax.scatter(test1[:, 0], test1[:, 1], color=colors1[1], s = 1, label=f'Cohort2')
confidence_ellipse(test1[:, 0], test1[:, 1], ax, facecolor=colors2[1], n_std=1.5, edgecolor=colors1[1])

ax.scatter(test2[:, 0], test2[:, 1], color=colors1[2], s = 1, label=f'Cohort3')
confidence_ellipse(test2[:, 0], test2[:, 1], ax, facecolor=colors2[2], n_std=1.5, edgecolor=colors1[2])

for spine in ax.spines.values():
    spine.set_visible(False)

x_start, x_end = np.min(all_data[:, 0]), np.max(all_data[:, 0])
y_start, y_end = np.min(all_data[:, 1]), np.max(all_data[:, 1])
x_mid = (x_end - x_start) / 2
y_mid = (y_end - y_start) / 2
 
ax.add_patch(FancyArrowPatch((x_start-8.3, y_start-8.7), (x_start-8.3, y_start+y_mid/2),
                              arrowstyle='->', mutation_scale=10, color='k'))
ax.add_patch(FancyArrowPatch((x_start-8.6, y_start-8.4), (x_start+x_mid/2-1, y_start-8.4),
                              arrowstyle='->', mutation_scale=10, color='k'))
#ax.add_patch(FancyArrowPatch((0, 0), (1, 0), transform=ax.transAxes, arrowstyle="->", color='k'))
# 添加y轴
#ax.add_patch(FancyArrowPatch((0, 0), (0, 1), transform=ax.transAxes, arrowstyle="->", color='k'))

# 移除刻度
ax.set_xticks([])
ax.set_yticks([])
# 设置图表限制，确保箭头不会超出图表范围
legend_handle1 = mlines.Line2D([], [], color=colors1[1], marker='o', linestyle='None',
                              markersize=10, label='Location (Central)')
legend_handle2 = mlines.Line2D([], [], color=colors1[2], marker='o', linestyle='None',
                              markersize=10, label='Location (Peripheral)')



#ax.set_xlim(-40, 30)
#ax.set_ylim(-40, 40)
plt.text(x_start-10, y_start-4, 'tSNE2', ha='center', rotation=90, fontsize=12)
plt.text(x_start, y_start-11, 'tSNE1', ha='center', fontsize=12)
#ax.set_xlabel("tSNE1")
#ax.set_ylabel("tSNE2")
plt.legend(handles=[legend_handle1, legend_handle2], loc='lower right', bbox_to_anchor=(1.2, 0), frameon=False, handletextpad=1)
plt.savefig('./lung_cancer.pdf', dpi=400, bbox_inches = 'tight')
plt.show()
