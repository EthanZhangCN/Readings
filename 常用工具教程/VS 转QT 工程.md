
使用CmakeLists生成的VS工程后，此时想把工程转换为Qt Creator工程，或者生成翻译文件的时候会发现

Qt的部分功能不可用，是灰色的
![](/images/2019-08-17-14-16-47.png)


此时只需要用记事本打开打开VS的工程文件
![](/images/2019-08-17-14-36-02.png)

找到关键字"Keyword"
![](/images/2019-08-17-14-36-44.png)

将内容更改为Qt4VSv1.0即可
![](/images/2019-08-17-14-38-05.png)

再打开工程，选中该工程再点击QtVSTools,发现多了个选项，选中Convert Project to VS Tools Project
![](/images/2019-08-17-14-39-40.png)

转换完后所有功能都亮了，再点生成Qt工程文件或者翻译件都可以了
![](/images/2019-08-17-14-42-54.png)