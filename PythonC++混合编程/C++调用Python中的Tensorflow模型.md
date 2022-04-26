---
title: C++调用Python中的Tensorflow模型
date: 2019/9/17 10:41:44
tags: CSDN迁移
---
 [ ](http://creativecommons.org/licenses/by-sa/4.0/) 版权声明：本文为博主原创文章，遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明。  本文链接：[https://blog.csdn.net/qq_17232031/article/details/83660935](https://blog.csdn.net/qq_17232031/article/details/83660935)   
    
  ### C++调用Tensorflow模型


    * [保存tensorflow模型](#tensorflow_2)
    * [模型加载代码](#_30)
    * [C++程序调用Python程序](#CPython_50)
    * [CMakeLists文件书写](#CMakeLists_101)
    * [结果](#_113)  
  
 利用c++调用Python2.7的程序，加载tensorflow模型（为什么不使用Python3，坑太多了，一直解决不好）。整个环境在Ubuntu16.04下完成，利用了kDevelop4 IDE编写C++程序，以及cmake文件。

 
## []()保存tensorflow模型

 首先利用Python写一段tensorflow保存模型的代码：

 
```
import tensorflow as tf
import os

def save_model_ckpt(ckpt_file_path):
    x = tf.placeholder(tf.int32,name='x')
    y = tf.placeholder(tf.int32,name='y')
    b = tf.Variable(1,name='b')
    xy = tf.multiply(x,y)
    op = tf.add(xy,b,name='op_to_store')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    path = os.path.dirname(os.path.abspath(ckpt_file_path))
    if os.path.isdir(path) is False:
        os.makedirs(path)

    tf.train.Saver().save(sess,ckpt_file_path)

    feed_dict = {x:4,y:3}
    print(sess.run(op,feed_dict))

save_model_ckpt('./model/model.ckpt')

```
 这会在model目录下回保存四个文件

 
## []()模型加载代码

 
```
#classify.py
import tensorflow as tf

def evaluate(pic):  
    sess = tf.Session()
    saver = tf.train.import_meta_graph('/home/tyl/Code/Kprojects/cpython/Test/model/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../model'))
    print(type(sess.run('b:0')))
    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')
    op = sess.graph.get_tensor_by_name('op_to_store:0')
    add_on_op = tf.multiply(op,2)
    ret = sess.run(add_on_op,{input_x:5,input_y:5})
    print ret
    sess.close()
    return pic

```
 这里要注意的是模型加载的路径一定要正确。。。。

 
## []()C++程序调用Python程序

 这里，利用C++程序调用模型加载的Python程序

 
```
//readTF.cpp
#include <Python.h>
#include <pythonrun.h>
#include <iostream>
#include <string.h>

int main()
{
  const int flag= 1;
  Py_Initialize();
  if (!Py_IsInitialized())
  {
    return -1;
  }

  PyRun_SimpleString("import sys");
  //路径一定要对
  PyRun_SimpleString("sys.path.append('/home/tyl/Code/Kprojects/cpython/Test')");
  
  PyObject* pMod = NULL;
  PyObject* pFunc = NULL;
  PyObject* pParm = NULL;
  PyObject* pRetVal = NULL;
  int iRetVal=999;
  PyObject* pName = PyString_FromString("classify");
  pMod = PyImport_Import(pName);//获取模块
  if (!pMod)
  {
	std::cout << pMod <<std::endl;
    return -1;
  }
  const char* funcName = "evaluate";
  pFunc = PyObject_GetAttrString(pMod,funcName);//获取函数
  if (!pFunc)
  {
    std::cout << "pFunc error" <<std::endl;
    return -1;
  }
  
  pParm = PyTuple_New(1);//新建元组
  PyTuple_SetItem(pParm, 0, Py_BuildValue("i",flag));//向Python模块传参
  pRetVal = PyObject_CallObject(pFunc,pParm);//获得返回结果

  PyArg_Parse(pRetVal,"i",&iRetVal);//解析成C++需要的形式
  std::cout<< iRetVal <<std::endl;
  return 0;
}

```
 
## []()CMakeLists文件书写

 
```
cmake_minimum_required(VERSION 2.6)
project(test)
set (CMAKE_BUILD_TYPE Debug)
set (CMAKE_CXX_FLAGS "-std=c++11")

include_directories( /usr/include/python2.7)		    
add_executable(readTF readTF.cpp)
target_link_libraries(readTF -lpython2.7)

```
 
## []()结果

 在KDevelop4上运行的结果  
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20181102201639537.png)

   
  