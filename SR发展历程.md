## SR发展历程
[9] SRCNN -->[47] ESPCN, [31] VDSR深度网络 --> [35]SRResNet残差块最大化残差学习 -->[38] EDSR移除BN层 --> [60] RDN, [58] RCAN, [1] CARN, [59] , [39], [55] LatticeNet

## 使用残差结构的网络
[31] VDSR ,[35] SRResNet, [38] EDSR

## 特征融合的网络
[60] , [37] , [38] EDSR

## 精巧设计loss公式
[35] SRResNet, [38] EDSR, [46]

## attention机制
[58] RCAN, [8] , [19] IMDN, [55] LatticeNet

## 轻量化结构
[23] , [43] , [56]

### benchmark suites
[22] , [45]

### 网络优化
#### 量化
[27] , [48]
#### 剪枝
[2] , [5]
#### 知识蒸馏
[42] , [54]
#### 轻量的网络
[20] IDN分组卷积, [19] IMDN, [1] CARN, [55] LatticeNet

## 最近的轻量SOTA
### 缺点是用很多卷积
[38] EDSR, [1] CARN, [20] IDN分组卷积
### 缺点是时间长的算子attention
[19] IMDN, [55] LatticeNet

## 轻量化网络的形式
### 显式：使用简单的算子减少模型复杂度
[32] , [52] , [37] , [1] , [20] IDN

[37] : 减少width和depth  
[32],[51]:递归结构  
[1],[20]:分组卷积

### 隐式：充分利用中间特征，提升网络区分能力，从而减少计算量，得到更好的结果
[34] , [1] , [52] , [19] , [55]

[34] : LapSRN利用每个pyramid层的特征  
[52] : MemNet使用gating机制用短时信息来连接长时信息  
[1]  : CARN采用瀑布机制整合局部、全局的特征  
[19] : IMDN采用contrast-aware channel attention 保留了partial 信息  
[55] : LatticeNet创造蝴蝶结构，也使用CCA动态结合两个残差快  

## 总结
### hierarchical特征融合
[1], [20] IDN, [19], [55] 会增加内存使用，因为移动设备的缓存有限  
### 注意力机制
[58], [28], [19], [55] 拥有不可容忍的算子推理时间，因为计算全局信息和按位相乘。  

### 图像感知领域用了量化技术的有
[18], [25], [50] ， 但是将量化用在SR领域可能会导致性能急剧降低。原因是：现在的架构移除了BN层，因为他们会导致SR有模糊的artifacts。但是移除BN层也导致了量化时的高动态变化范围。  
[40], [36] 就是为了解决上面这些问题。  



## 参考文献
[1] Namhyuk Ahn, Byungkon Kang, and Kyung-Ah Sohn. **Fast,accurate, and lightweight super-resolution with cascading residual network**. In ECCV (10), volume 11214 of Lecture Notes in Computer Science, pages 256–272. Springer, 2018.   

[5] S. K. Chao, Z. Wang, Y. Xing, and G. Cheng. **Directional pruning of deep neural networks**. 2020.

[8] Tao Dai, Jianrui Cai, Yongbing Zhang, Shu Tao Xia, and Lei Zhang. **Second-order attention network for single image super-resolution**  In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019

[9] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. **Learning a deep convolutional network for image super-resolution**. In ECCV (4), volume 8692 of Lecture Notes in Computer Science, pages 184–199. Springer, 2014.

[18] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko,W.Wang, T. Weyand, M. Andreetto, and H. Adam. **Mobilenets: Efficient convolutional neural networks for mobile vision applications**. 2017.  

[19] Zheng Hui, Xinbo Gao, Yunchu Yang, and Xiumei Wang. **Lightweight image super-resolution with information multidistillation network**. In ACM Multimedia, pages 2024–2032. ACM, 2019.  

[20] Zheng Hui, Xiumei Wang, and Xinbo Gao. **Fast and accurate single image super-resolution via information distillation network**. In CVPR, pages 723–731. IEEE Computer Society, 2018.

[22] A. Ignatov, R. Timofte, A. Kulik, S. Yang, K. Wang, F. Baum, M. Wu, L. Xu, and L. Van Gool. **Ai benchmark:All about deep learning on smartphones in 2019**. In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pages 3617–3635, 2019.  

[23] A. Ignatov, R. Timofte, T. V. Vu, T. M. Luu, and C. Jung. **Pirm challenge on perceptual image enhancement on smartphones:Report**. Springer, Cham, 2018.  

[25] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko. **Quantization and training of neural networks for efficient integer-arithmetic-only inference**. 2017.  

[27] K. Jia and M. Rinard. **Efficient exact verification of binarized neural networks**. 2020.

[28] Jie, Hu, Li, Shen, Samuel, Albanie, Gang, Sun, Enhua, and Wu. **Squeeze-and-excitation networks**. IEEE transactions on pattern analysis and machine intelligence, 2019.

[31] Jiwon Kim, Jung Kwon Lee, and Kyoung Mu Lee. **Accurate image super-resolution using very deep convolutional networks**. In CVPR, pages 1646–1654. IEEE Computer Society, 2016.

[32] Jiwon Kim, Jung Kwon Lee, and Kyoung Mu Lee. **Deeply recursive convolutional network for image super-resolution**. In CVPR, pages 1637–1645. IEEE Computer Society, 2016.

[34] Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang. **Deep laplacian pyramid networks for fast and accurate super-resolution**. In CVPR, pages 5835–5843. IEEE Computer Society, 2017.  

[35] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, and Zehan andWang. **Photorealistic single image super-resolution using a generative adversarial network**. 2017.

[36] H. Li, C. Yan, S. Lin, X. Zheng, B. Zhang, F. Yang, and R. Ji. Pams: **Quantized super-resolution via parameterized max scale**. In Springer, Cham, 2020.  

[37] Zhen Li, Jinglei Yang, Zheng Liu, Xiaomin Yang, Gwanggil Jeon, and Wei Wu. **Feedback network for image superresolution**. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.  

[38] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. **Enhanced deep residual networks for single image super-resolution**. 2017.

[39] Jie Liu,Wenjie Zhang, Yuting Tang, Jie Tang, and Gangshan Wu. **Residual feature aggregation network for image superresolution**. pages 2356–2365, 06 2020.

[40] Y. Ma, H. Xiong, Z. Hu, and L. Ma. **Efficient super resolution using binarized neural network**. 2018.

[42] H. Mobahi, M. Farajtabar, and P. L. Bartlett. **Self-distillation amplifies regularization in hilbert space**. 2020.

[43] S. Nah, S. Son, R. Timofte, K. M. Lee, and T. Huck. **Ntire 2020 challenge on image and video deblurring**. IEEE, 2020.

[45] V. J. Reddi, C. Cheng, D. Kanter, P. Mattson, and G. Schmuelling. **Mlperf inference benchmark**. In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), pages 446–459, 2020. 

[46] Mehdi S. M. Sajjadi, Bernhard Scholkopf, and Michael Hirsch. **Enhancenet: Single image super-resolution through automated texture synthesis**. In IEEE International Conference on Computer Vision, 2017.

[47] Wenzhe Shi, Jose Caballero, Ferenc Husz´ar, Johannes Totz, and Zehan Wang. **Real-time single image and video superresolution using an efficient sub-pixel convolutional neural network**. 2016.

[48] M. Shkolnik, B. Chmiel, R. Banner, G. Shomron, Y. Nahshan, A. Bronstein, and U. Weiser. **Robust quantization: One model to rule them all**. 2020.


[50] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. **Rethinking the inception architecture for computer vision**. IEEE, pages 2818–2826, 2016.

[51] Ying Tai, Jian Yang, and Xiaoming Liu. **Image superresolution via deep recursive residual network**. In CVPR, pages 2790–2798. IEEE Computer Society, 2017. 

[52] Ying Tai, Jian Yang, Xiaoming Liu, and Chunyan Xu. **Memnet:A persistent memory network for image restoration**. In ICCV, pages 4549–4557. IEEE Computer Society, 2017.  

[54] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou. **Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers**. 2020.  

[55] Luo Xiaotong, Xie Yuan, Zhang Yulun, Qu Yanyun, Li Cuihua, and Fu Yun. **Latticenet: Towards lightweight image super-resolution with lattice block**. 2020. 

[56] Kai Zhang and Nan Nan. **AIM 2019 challenge on constrained super-resolution: Methods and results**. In 2019 IEEE/CVF International Conference on Computer Vision Workshops, ICCV Workshops 2019, Seoul, Korea (South), October 27-28, 2019, pages 3565–3574. IEEE, 2019. 

[58] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu. **Image super-resolution using very deep residual channel attention networks**. 2018.  

[59] Yulun Zhang, Kunpeng Li, Kai Li, Bineng Zhong, and Yun Fu. **Residual non-local attention networks for image restoration**. 2019.

[60] Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, and Yun Fu. **Residual dense network for image super-resolution**. 2018.
