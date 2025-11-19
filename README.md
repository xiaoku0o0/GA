此处提供GA演示视频的全部manim源码，使用者可免费且无条件地使用所有开源内容。

[![cc0][cc0-image]][cc0]

[cc0]: https://creativecommons.org/public-domain/cc0/
[cc0-image]: https://licensebuttons.net/p/zero/1.0/88x31.png

[![Python](https://camo.githubusercontent.com/36a52016e02020b1b2b3a4b07812957a13bf404e03a8793f1793415a6a40be22/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d76332e31312d677265656e2e7376673f7374796c653d666c6174)](https://www.python.org/) [![manim](https://camo.githubusercontent.com/5d142d7c8431408522b6a907e828d11de85f9ff8e0d679b134dc5e3af1319886/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6d616e696d2d76302e31382e302d677265656e2e7376673f7374796c653d666c6174)](https://github.com/3b1b/manim)

## 文件说明

manim文件夹提供了使用manim制作的分镜头的python源码，文件名参考分镜设计稿.pdf

GA_code文件夹提供了遗传算法的python、matlab代码。GA_1dim提供了一维GA，GA_ndim提供了多变量遗传算法，同时支持添加约束条件。



## 渲染命令

```python
manim BM1.py BM1
```

其中，BM1.py为文件名，BM1为类名

亦可在该文件下补充以下代码，直接运行文件：

```python
if __name__ == '__main__':
	import os
    os.system('manim MA1.py MA1')
```



