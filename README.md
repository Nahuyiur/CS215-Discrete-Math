# 基于图论的图像降噪方法及加速算法研究

## 📖 项目简介

本项目旨在研究 **基于图论的图像降噪方法**，通过构建图拉普拉斯矩阵和加权邻接矩阵，将图像建模为加权图，以利用图的平滑特性实现噪声抑制。同时，结合 **预条件共轭梯度法（PCG）** 和 **Nesterov 加速法**，优化计算效率。此外，探索了 **基于图卷积网络（GCN）** 的图像降噪方法，尝试结合图论的理论优势和深度学习的强大特性。

---

## ✨ 核心内容

### 1. **噪声建模与处理**
#### 常见噪声类型
- **高斯噪声**：像素值服从正态分布，适用于模拟热噪声。
- **椒盐噪声**：离散黑白点噪声，适合中值滤波处理。
- **均匀噪声**：像素值均匀分布的噪声，多用于量化误差模拟。
- **泊松噪声**：与信号强度相关的统计噪声。
- **瑞利噪声** 和 **伽马噪声**：多应用于医学成像和雷达图像处理。

#### 去噪方法
- **传统滤波**：
  - 高斯滤波：适用于高斯噪声，平滑图像但可能模糊边缘。
  - 中值滤波：适合去除椒盐噪声，能很好地保留边缘信息。
  - 双边滤波：结合空间距离和像素值差异，平滑噪声并保留边缘。

实验结果：
| 噪声类型     | 最优滤波方法  | 平均 PSNR (dB) | 平均 SSIM |
|--------------|---------------|----------------|-----------|
| 高斯噪声     | 高斯滤波      | 28.50          | 0.31      |
| 椒盐噪声     | 中值滤波      | 30.36          | 0.35      |
| 均匀噪声     | 均值滤波      | 30.71          | 0.66      |
| 泊松噪声     | 高斯滤波      | 32.31          | 0.80      |
| 瑞利噪声     | 中值滤波      | 29.72          | 0.51      |
| 伽马噪声     | 双边滤波      | 28.93          | 0.43      |

---

### 2. **基于图论的降噪方法**
#### 图模型与随机游走
- 使用 **图拉普拉斯矩阵** 和 **加权邻接矩阵** 描述图像像素关系。
- 通过 **随机游走算法** 平滑噪声，模拟像素间的相互作用。

#### 图割算法
- 将图像降噪任务建模为图的最小割问题。
- 采用 **Boykov-Kolmogorov 算法** 优化计算效率。
- 对于某些噪声，降噪效果一般，但提供了独特的优化视角。

实验结果（伽马噪声降噪）：
| 评价指标      | 彩色图像         | 灰度图像         |
|---------------|------------------|------------------|
| MSE           | 47.17           | 103.71          |
| PSNR (dB)     | 31.37           | 27.97           |
| SSIM          | 0.54            | 0.42            |
| 执行时间 (s)  | 13.7            | 12.1            |

---

### 3. **加速算法**
#### 方法与优化
- **预条件共轭梯度法（PCG）**：
  - 引入预条件器改善矩阵条件数，加速收敛。
- **Nesterov 加速法**：
  - 利用动量项提高更新效率。

#### 加速效果
实验结果（双边滤波加速对比）：
| 方法              | 执行时间（ms） | 加速效果 |
|-------------------|----------------|----------|
| 普通双边滤波      | 450~480        | 1x       |
| PCG 加速          | 210~225        | 2-3x     |
| Nesterov 加速      | 185~192        | 3-4x     |

---

### 4. **基于图卷积网络的探索**
- 构建两层 **GCN 模型**，尝试结合图论与深度学习特性。
- 虽未达到理想效果，但为未来研究提供了新思路。

实验结果：
| 噪声类型     | 强度      | PSNR (dB) | SSIM  |
|--------------|-----------|-----------|-------|
| 高斯噪声     | σ=20      | 20.76     | 0.30  |
| 椒盐噪声     | 密度=0.05 | 21.87     | 0.28  |

---

## 📄 项目简介

本项目旨在研究基于图论的图像降噪方法及其加速策略。通过构建图拉普拉斯矩阵和加权邻接矩阵，将图像建模为加权图，从而利用图的平滑特性实现噪声抑制。此外，为了应对传统方法迭代次数多、计算效率低的不足，我们引入了预条件共轭梯度法（PCG）和 Nesterov 加速法，并探索了基于图卷积网络（GCN）的降噪方法。

实验结果表明：
- 传统基于图论的方法在滤波性能上表现良好，但计算效率较低。
- PCG 和 Nesterov 加速方法在保证滤波效果的同时实现了 2-4 倍的加速。
- GCN 在降噪性能上具有潜力，但受限于模型复杂度和数据规模。

本项目的主要模块包括：
- **噪声建模**：模拟高斯噪声、椒盐噪声、泊松噪声等常见噪声类型。
- **图像滤波**：实现均值滤波、高斯滤波、中值滤波、双边滤波等常用方法。
- **基于图论的方法**：随机游走降噪、图割降噪。
- **加速算法**：PCG 和 Nesterov 加速方法。
- **基于 GCN 的方法**：使用图卷积网络对图像进行降噪。

所有代码和实验数据均包含在本项目中。

---

## 🛠 安装说明

使用以下步骤快速设置环境并安装所有依赖：

### 环境配置

1. **创建一个新的 Conda 环境**：
   ```bash
   conda create -n graph-denoise python=3.8 -y
   ```

2. **激活环境**：
   ```bash
   conda activate graph-denoise
   ```

3. **安装依赖库**：
   ```bash
   conda install -c conda-forge numpy scipy matplotlib networkx scikit-image tqdm -y
   ```

4. **安装 PyTorch（可选，用于 GCN 部分）**：
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

5. **安装 JupyterLab（可选，用于交互式开发）**：
   ```bash
   conda install -c conda-forge jupyterlab
   ```

6. **验证安装成功**：
   ```bash
   python -c "import numpy; import scipy; import matplotlib; import networkx; import skimage; import torch"
   ```


至此，环境配置完成，即可运行项目中的代码。

----

## 📊 实验结果

实验结果显示：
- **传统滤波方法**：对于简单噪声（如高斯噪声），高斯滤波效果最佳；对于椒盐噪声，中值滤波性能最优。
- **基于图论的方法**：随机游走和图割方法在保留边缘的同时显著降低噪声，但计算复杂度较高。
- **加速算法**：PCG 和 Nesterov 方法在滤波性能几乎不受影响的情况下实现了显著的计算加速。
- **基于 GCN 的方法**：在噪声强度较高的情况下，GCN 对复杂噪声的处理能力仍需进一步优化。

详细实验结果请参考项目文档及代码注释。

-----
## 🔗 参考文献

- **均值滤波、高斯滤波、中值滤波、双边滤波4者有什么区别呢？应用场合有什么区别呢？**  
  Available at: [CSDN](https://blog.csdn.net/weixin_43501408/article/details/142760150)  
  *Accessed on: January 12, 2025*

- **OpenCV 学习：9 双边滤波 bilateralFilter**  
  Available at: [Zhihu](https://zhuanlan.zhihu.com/p/127023952)  
  *Accessed on: January 12, 2025*

- **图像噪声的特点以及分类（一）**  
  Available at: [CSDN](https://blog.csdn.net/qq_27825451/article/details/102923053)  
  *Accessed on: January 12, 2025*

- **A Review Paper: Noise Models in Digital Image Processing**  
  *arXiv*  
  Available at: [arXiv](https://arxiv.org/pdf/1505.03489)  
  *2015*

- **Benchmarking Deep Learning-Based Low-Dose CT Image Denoising Algorithms**  
  Available at: [arXiv](https://info.arxiv.org/help/license/index.html#licenses-available)  
  *Accessed on: January 12, 2025*

- **Total Variation Image Denoising Algorithm Based on Graph Cut**  
  Authors: WU Ya-dong, SUN Shi-xin, ZHANG Hong-ying, HAN Yong-guo, and Chen Bo  
  *Acta Electronica Sinica*, Volume 35, Issue 2, Pages 265–268, 2007

- **随机游走 (Random Walk) 算法**  
  Available at: [CSDN](https://blog.csdn.net/qq_43186282/article/details/114585885)  
  *Accessed on: January 12, 2025*

- **An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision**  
  Authors: Yuri Boykov and Vladimir Kolmogorov  
  *IEEE Transactions on Pattern Analysis and Machine Intelligence*, Volume 26, Issue 9, Pages 1124–1137, 2004  
  Available at: [IEEE](https://ieeexplore.ieee.org/document/1316848)

- **Accelerated graph-based nonlinear denoising filters**  
  *arXiv*  
  Available at: [arXiv](https://arxiv.org/pdf/1512.00389)  
  *2015*

- **图卷积网络 (Graph Convolutional Networks, GCN) 详细介绍**  
  Available at: [CSDN](https://blog.csdn.net/qq_43787862/article/details/113830925)  
  *Accessed on: January 12, 2025*

- **Semi-Supervised Classification with Graph Convolutional Networks**  
  Authors: Thomas Kipf and Max Welling  
  *arXiv*, 2016  
  Available at: [arXiv](https://arxiv.org/pdf/1609.02907)

  ---
## 📄 问题与建议

非常感谢您对本项目的关注！如果您有任何问题、建议，或者发现可以改进的地方，欢迎随时与我们联系。我们非常期待您的宝贵意见和建设性反馈，以共同完善和优化本项目。

您可以直接在仓库中提交问题（issue）或拉取请求（pull request）。让我们携手提升项目的质量与功能！



