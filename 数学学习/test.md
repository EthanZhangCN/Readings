图像质量评价指标之 PSNR 和 SSIM
###1. PSNR (Peak Signal-to-Noise Ratio) 峰值信噪比
给定一个大小为 m×n 的干净图像 I 和噪声图像 K，均方误差 (MSE) 定义为：
$$
MSE=\frac{1}{mn}\displaystyle \sum^{m-1}_{i=0} \displaystyle \sum^{n-1}_{j=0}[I(i,j)-K(i,j)]^2
$$
然后 PSNR(dB) 就定义为：
$$
PSNR=10⋅log_{10}(\frac{MAX_I^2}{IMSE})
$$
其中 $MAX^2_I 为图片可能的最大像素值。如果每个像素都由 8 位二进制来表示，那么就为 255。通常，如果像素值由 B 位二进制来表示，那么$$MAX_I=2^B−1$。

一般地，针对 uint8 数据，最大像素值为 255,；针对浮点型数据，最大像素值为 1。

上面是针对灰度图像的计算方法，如果是彩色图像，通常有三种方法来计算。

分别计算 RGB 三个通道的 PSNR，然后取平均值。
计算 RGB 三通道的 MSE ，然后再除以 3 。
将图片转化为 YCbCr 格式，然后只计算 Y 分量也就是亮度分量的 PSNR。
其中，第二和第三种方法比较常见。
```
# im1 和 im2 都为灰度图像，uint8 类型

# method 1
diff = im1 - im2
mse = np.mean(np.square(diff))
psnr = 10 * np.log10(255 * 255 / mse)

# method 2
psnr = skimage.measure.compare_psnr(im1, im2, 255)
```
compare_psnr(im_true, im_test, data_range=None) 函数原型可见此处

针对超光谱图像，我们需要针对不同波段分别计算 PSNR，然后取平均值，这个指标称为 MPSNR。

###2. SSIM (Structural SIMilarity) 结构相似性
SSIM 公式基于样本 x 和 y 之间的三个比较衡量：亮度 (luminance)、对比度 (contrast) 和结构 (structure)。
$$
l(x,y)=\frac{2μ_xμ_y+c_1}{μ^2_x+μ^2_y+c_1}
$$
$$
c(x,y)=\frac{2σ_xσ_y+c_2}{σ^2_x+σ^2_y+c_2}
$$
$$
s(x,y)=\frac {σ_{xy}+c_3}{σ_xσ_y+c_3}
$$
一般取 $c_3=\frac{c_2}{2}$。

$μ_x 为 x 的均值$
$μ_y 为 y 的均值$
$σ^2_x 为 x 的方差$
$σ^2_y 为 y 的方差$
$σ_{xy} 为 x 和 y 的协方差$
$c_1=(k_1L)^2,c_2=(k_2L)^2 为两个常数，避免除零$
$L 为像素值的范围，2^B−1$
$k_1=0.01,k_2=0.03 为默认值$
那么
$$
SSIM(x,y)=[l(x,y)^α⋅c(x,y)^β⋅s(x,y)^γ]
$$
将 α,β,γ 设为 1，可以得到
$$
SSIM(x,y)=\frac{(2μ_xμ_y+c_1)(2σ_{xy}+c_2)}{(μ^2_x+μ^2_y+c_1)(σ^2_x+σ^2_y+c^2)}
$$
每次计算的时候都从图片上取一个 N×N 的窗口，然后不断滑动窗口进行计算，最后取平均值作为全局的 SSIM。
```
# im1 和 im2 都为灰度图像，uint8 类型
ssim = skimage.measure.compare_ssim(im1, im2, data_range=255)
```
compare_ssim(X, Y, win_size=None, gradient=False, data_range=None, multichannel=False, gaussian_weights=False, full=False, **kwargs) 函数原型可见此处
https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim
针对超光谱图像，我们需要针对不同波段分别计算 SSIM，然后取平均值，这个指标称为 MSSIM。