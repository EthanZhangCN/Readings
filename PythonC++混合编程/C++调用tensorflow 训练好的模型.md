---
title: C++调用tensorflow 训练好的模型
date: 2019/9/17 10:40:21
tags: CSDN迁移
---
 [ ](http://creativecommons.org/licenses/by-sa/4.0/) 版权声明：本文为博主原创文章，遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明。  本文链接：[https://blog.csdn.net/Pennypinch/article/details/78342948](https://blog.csdn.net/Pennypinch/article/details/78342948)   
    
   分享给大家，希望可以帮助到大家.

 

 我看到有些说只能安装32位的py,我开始也是这样的，但是安装TensorFlow做测试的时候，就一直有问题，所以呀，我就换成了ananconda 

 ![](https://img-blog.csdn.net/20171025162612454?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvUGVubnlwaW5jaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

 安装这个网上一大堆，自己可以好好看看哦！

 

 然后就是新建一个C++的工程

 

 1.把ananconda的减压后，将里面的inlcude和libs两个文件夹拷贝到sln的同一级目录下

 

 2.然后打开libs，复制一份python35.lib，并命名为python35_d.lib

 

 3.C++->常规->附加包含目录，输入..\include;

 

 4.链接器->常规->附加目录项，输入..\libs;

 

 5.链接器->输入->附加依赖项，添加python35_d.lib;

 

 6. python35.dll拷贝到Debug目录下(与Test.exe同目录)

 

 7.将py拷贝到Debug目录下(与Test.exe同目录)

 

 8.将你训练好的模型新建一个文件夹拷贝到C++项目文件夹里来

 ![](https://img-blog.csdn.net/20171025163040166?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvUGVubnlwaW5jaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

 1-----测试图片 2.3就是py里面的东西 4就是你的模型

 

 好了 我开始贴代码了~~~

 C++：

 

 
```
 void testImage(char * path)
{
	try{
		Py_Initialize();
		PyEval_InitThreads();
		PyObject*pFunc = NULL;
		PyObject*pArg = NULL;
		PyObject* module = NULL;
		module = PyImport_ImportModule("myModel");//myModel:Python文件名
		if (!module) {
			printf("cannot open module!");
			//Py_Finalize();
		}
		pFunc = PyObject_GetAttrString(module, "test_one_image");//test_one_image:Python文件中的函数名
		if (!pFunc) {
			printf("cannot open FUNC!");
			//Py_Finalize();
		}
		//开始调用model
		pArg = Py_BuildValue("(s)", path);
		if (module != NULL) {
			PyGILState_STATE gstate;
		        gstate = PyGILState_Ensure();
			 PyEval_CallObject(pFunc, pArg);
			PyGILState_Release(gstate);
	
		}
	}
	 catch (exception& e)
	 {
		 cout << "Standard exception: " << e.what() << endl;
	 }
}
```
 python:

 

 

 
```
 def test_one_image(test_dir):
    image = Image.open(test_dir)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image = image.resize([32, 32])
    image_array = np.array(image)

    with tf.Graph().as_default():
        image = tf.cast(image_array, tf.float32)
        image = tf.reshape(image, [1, 32, 32, 3])#调整image的形状
        p = mmodel(image, 1)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32, shape=[32, 32, 3])
        saver = tf.train.Saver()
        model_path='E:/MyProject/MachineLearning/call64PY/test/model/'
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, tf.train.latest_checkpoint('E:/MyProject/MachineLearning/call64PY/test/model/'))
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('加载ckpt成功！')
            else:
                print('error')

            prediction = sess.run(logits, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('case0： %.6f' % prediction[:, 0])
                return result
            else:
                print('-case1： %.6f' % prediction[:, 1])
                return result2
```
 这里面好多坑啊~~~

 

 

 

 

 

 

   
 