import numpy as np
import matplotlib.pyplot as plt


# 定義條件分佈函數
def sample_x1_given_x2(x2, mean1=0, mean2=0, sigma1=1, sigma2=1, rho=0.8):
    """給定 x2，抽樣 x1"""
    conditional_mean = mean1 + rho * (sigma1 / sigma2) * (x2 - mean2)
    conditional_std = np.sqrt(1 - rho**2) * sigma1
    return np.random.normal(conditional_mean, conditional_std)


def sample_x2_given_x1(x1, mean1=0, mean2=0, sigma1=1, sigma2=1, rho=0.8):
    """給定 x1，抽樣 x2"""
    conditional_mean = mean2 + rho * (sigma2 / sigma1) * (x1 - mean1)
    conditional_std = np.sqrt(1 - rho**2) * sigma2
    return np.random.normal(conditional_mean, conditional_std)


# Gibbs 採樣器
def gibbs_sampler(n_samples, burn_in=100000):
    """生成 n_samples 個 (x1, x2) 樣本"""
    samples = []
    x1, x2 = 0, 0  # 初始值

    for i in range(n_samples + burn_in):
        x1 = sample_x1_given_x2(x2)
        x2 = sample_x2_given_x1(x1)

        if i >= burn_in:
            samples.append((x1, x2))

    return np.array(samples)


# 設定樣本數
n_samples = 100000
samples = gibbs_sampler(n_samples)

# 繪製 2D 直方圖
plt.hist2d(samples[:, 0], samples[:, 1], bins=100, cmap="viridis")
plt.colorbar(label="Density")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("2D Histogram of Samples from Gibbs Sampler")
plt.show()
