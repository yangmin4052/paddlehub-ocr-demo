# PaddleHub OCR识别

## 项目背景介绍

使用ORC协助商户有效管理商品

## 模型介绍

- 本例采用paddlehub工具组, 模型使用chinese_ocr_db_crnn_server, 识别文字算法采用CRNN（Convolutional Recurrent Neural Network）即卷积递归神经网络

## 模型使用

### 环境准备

```py
!pip install --upgrade paddlehub -i https://mirror.baidu.com/pypi/simple
```

```py
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
```

### 定义待识别照片

```py
# 待预测图片
test_img_path = "./test.jpg"

img = mpimg.imread(test_img_path) 

# 展示待预测图片
plt.figure(figsize=(10,10))
plt.imshow(img) 
plt.axis('off') 
plt.show()
```

```py
!hub install chinese_ocr_db_crnn_server==1.1.2
```

### 预测文字

```py
import paddlehub as hub
import cv2

test_img_path = "./test.jpg"
ocr = hub.Module(name="chinese_ocr_db_crnn_server")
results = ocr.recognize_text(images=[cv2.imread(test_img_path)])
```

### 展示结果

```py
plt.imshow(img)
ax = plt.gca()
for result in results:
    # print(result['data'])
    for i,item in enumerate(result['data']):
        print(i, item['text'], item['confidence'], item['text_box_position'])
        left_top=item['text_box_position'][0]
        right_bottom=item['text_box_position'][2]
        w=right_bottom[0]-left_top[0]
        h=right_bottom[1]-left_top[1]
        ax.add_patch(plt.Rectangle(left_top, w, h, color="blue", fill=False, linewidth=1))
        ax.text(*left_top, i, bbox={'facecolor':'blue', 'alpha':0.5})
plt.savefig("./a.jpg")
plt.show()
```

## 总结与升华

初步了解了paddlehub使用

## 个人总结

个人对cv比较感兴趣,道阻且长
