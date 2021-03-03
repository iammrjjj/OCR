# 基于CNN的验证码识别

====
### 其他语言： [English](https://github.com/PatrickLib/captcha_recognize/blob/master/README.md) [中文](https://github.com/PatrickLib/captcha_recognize/blob/master/README-zhcn.md)

基于TensorFlow的验证码识别，运行环境 OSX 10.15.7，CPU，Python 2.7

验证准确率为86.1%，训练集大小为50000，20000轮训练

依赖环境
=======
- python 2.7
- TensorFlow 1.1
- captcha

使用步骤
=======
## 1.准备验证码图片

```
python captcha_gen_default.py
```

## 2.将验证码图片转换为tfrecords格式

```
python captcha_records.py
```

## 3.模型训练

```
python captcha_train.py
```

## 4.模型评估
```
python captcha_eval.py
```

## 5.验证码识别

```
python captcha_recognize.py
```
结果如下
```
...
image ib1B_num3513.png recognize ----> 'ib1B'
image nhKV_num3144.png recognize ----> 'nhKV'
image 7CJB_num1213.png recognize ----> '7CJB'
image DenZ_num2848.png recognize ----> 'DenZ'
image JPFZ_num1849.png recognize ----> 'JPFZ'
```

Ref
===
PatrickLib/captcha_recognize
