---
title: 哈尔小波变换的原理及其实现(Haar)
date: 2019/9/20 14:51:44
tags: CSDN迁移
---
   Haar小波在图像处理和数字水印等方面应用较多，这里简单的介绍一下哈尔小波的基本原理以及其实现情况。

 一、Haar小波的基本原理

 数学理论方面的东西我也不是很熟悉，这边主要用简单的例子来介绍下Haar小波的使用情况。

 例如：有a=[8,7,6,9]四个数，并使用b[4]数组来保存结果.

 则一级Haar小波变换的结果为:

 b[0]=(a[0]+a[1])/2, b[2]=(a[0]-a[1])/2

 b[1]=(a[2]+a[3])/2, b[3]=(a[2]-a[3])/2

 即依次从数组中取两个数字，计算它们的和以及差，并将和一半和差的一半依次保存在数组的前半部分和后半部分。

 例如：有a[8]，要进行一维Haar小波变换，结果保存在b[8]中

 则一级Haar小波变换的结果为:

 b[0]=(a[0]+a[1])/2, b[4]=(a[0]-a[1])/2

 b[1]=(a[2]+a[3])/2, b[5]=(a[2]-a[3])/2

 b[2]=(a[4]+a[5])/2, b[6]=(a[4-a[5]])/2

 b[3]=(a[6]+a[7])/2, b[7]=(a[6]-a[7])/2

 如果需要进行二级Haar小波变换的时候，只需要对b[0]-b[3]进行Haar小波变换.

 对于二维的矩阵来讲，每一级Haar小波变换需要先后进行水平方向和竖直方向上的两次一维小波变换,行和列的先后次序对结果不影响。

 二、Haar小波的实现

 使用opencv来读取图片及像素，对图像的第一个8*8的矩阵做了一级小波变换

 **[cpp]** [view plain](http://blog.csdn.net/augusdi/article/details/8680011#)[copy](http://blog.csdn.net/augusdi/article/details/8680011#)   
   
   
 
  1. #include <cv.h> 
  2. #include <highgui.h> 
  3. #include <iostream> 
  4. using namespace std;int main() 
  5. { 
  6. IplImage* srcImg; 
  7. double imgData[8][8]; 
  8. int i,j; 
  9. 
  10. srcImg=cvLoadImage("lena.bmp",0); 
  11. 
  12. cout<<"原8*8数据"<<endl; 
  13. for( i=0;i<8;i++) 
  14. { 
  15. for( j=0;j<8;j++) 
  16. { 
  17. imgData[i][j]=cvGetReal2D(srcImg,i+256,j+16); 
  18. cout<<imgData[i][j]<<" "; 
  19. } 
  20. cout<<endl; 
  21. } double tempData[8]; 
  22. //行小波分解 
  23. for( i=0;i<8;i++) 
  24. { 
  25. for( j=0;j<4;j++) 
  26. { 
  27. double temp1=imgData[i][2*j]; 
  28. double temp2=imgData[i][2*j+1]; 
  29. tempData[j]=(temp1+temp2)/2; 
  30. tempData[j+4]=(temp1-temp2)/2; 
  31. } for( j=0;j<8;j++) 
  32. { 
  33. imgData[i][j]=tempData[j]; 
  34. } 
  35. } //列小波分解 
  36. for( i=0;i<8;i++) 
  37. { 
  38. for( j=0;j<4;j++) 
  39. { 
  40. double temp1=imgData[2*j][i]; 
  41. double temp2=imgData[2*j+1][i]; 
  42. tempData[j]=(temp1+temp2)/2; 
  43. tempData[j+4]=(temp1-temp2)/2; 
  44. } 
  45. for( j=0;j<8;j++) 
  46. { 
  47. imgData[j][i]=tempData[j]; 
  48. } 
  49. } 
  50. cout<<"1级小波分解数据"<<endl; 
  51. for( i=0;i<8;i++) 
  52. { 
  53. for( j=0;j<8;j++) 
  54. { 
  55. cout<<imgData[i][j]<<" "; 
  56. } 
  57. cout<<endl; 
  58. } 
  59. //列小波逆分解 
  60. for( i=0;i<8;i++) 
  61. { 
  62. for( j=0;j<4;j++) 
  63. { 
  64. double temp1=imgData[j][i]; 
  65. double temp2=imgData[j+4][i]; 
  66. tempData[2*j]=temp1+temp2; 
  67. tempData[2*j+1]=temp1-temp2; 
  68. } for( j=0;j<8;j++) 
  69. { 
  70. imgData[j][i]=tempData[j]; 
  71. } 
  72. } //行小波逆分解 
  73. for( i=0;i<8;i++) 
  74. { 
  75. for( j=0;j<4;j++) 
  76. { 
  77. double temp1=imgData[i][j]; 
  78. double temp2=imgData[i][j+4]; 
  79. tempData[2*j]=temp1+temp2; 
  80. tempData[2*j+1]=temp1-temp2; 
  81. } 
  82. for( j=0;j<2*4;j++) 
  83. { 
  84. imgData[i][j]=tempData[j]; 
  85. } 
  86. } cout<<"1级小波逆分解数据"<<endl; 
  87. for( i=0;i<8;i++) 
  88. { 
  89. for( j=0;j<8;j++) 
  90. { 
  91. cout<<imgData[i][j]<<" "; 
  92. } 
  93. cout<<endl; 
  94. } 
  95. 
  96. return 0; 
  97. }   
   
 