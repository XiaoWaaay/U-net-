import numpy
import numpy as np
import torch
import torch.optim
import cv2
import os
import time
from PIL import Image
import matplotlib.pyplot as plt

import Unet_principle as Unet


def dehaze_image(image_name):
    data_hazy = Image.open(image_name)
    data_hazy = np.array(data_hazy) / 255.0
    original_img = data_hazy.copy() # 未去雾前的图片
    data_hazy = torch.from_numpy(data_hazy).float()     # 将数组转化成张量
    data_hazy = data_hazy.permute(2, 0, 1)              # 维度转化
    data_hazy = data_hazy.unsqueeze(0)                  # 升维度 方便处理数据

    dehaze_net = Unet.Unet(3,3)     # dehaze_net生成的一个net模型
    dehaze_net.eval()   #! 修改
    dehaze_net.load_state_dict(torch.load('saved_models_bilinear/dehaze_net_epoch_29.pth', map_location=torch.device('cpu')))   # dehaze_net将dehazer.pth中的参数加载进来
    #dehaze_net = torch.load('saved_models/dehaze_net_epoch_6.pth', map_location=torch.device('cpu'))

    clean_image = dehaze_net(data_hazy).detach().numpy().squeeze()
    clean_image = np.swapaxes(clean_image, 0, 1)
    clean_image = np.swapaxes(clean_image, 1, 2)

    # 展示未去雾前的图片
    plt.subplot(1, 2, 1)    # 1行2列 此时位于第1列
    plt.imshow(original_img)  # 将数组的值以图片形式展示出来
    plt.axis('off')         # 不显示坐标轴
    plt.title('Original Image') # 图片名称
    # 展示去雾后的图片
    plt.subplot(1, 2, 2)    # 1行2列 此时位于第2列
    plt.imshow(clean_image)
    plt.axis('off')
    plt.title('Dehaze Image')
    plt.savefig(f'./results_compare/dehaze_{i}.png')   # 保存对比图
    plt.show()

    return clean_image


if __name__ == '__main__':
    a=0
    for i in range(1, 30):  # 控制去雾图片的数量

        root_dir = 'test_image/'
        img_name =  'GT_'+str(i)+'.jpg'

        img = os.path.join(root_dir, img_name)
        start = time.process_time() # 去雾开始时间
        img_res = dehaze_image(img) # 单图片去雾
        end = time.process_time()   # 去雾结束时间
        b = end - start     # 去雾所用时间
        a = a + b           # 所有图片去雾所用时间

        img_trans = cv2.cvtColor(numpy.asarray(img_res), cv2.COLOR_RGB2BGR) # 颜色空间转换函数 RGB->BGR
        cv2.imwrite("./results_dehaze/dehaze_" + str(i) + ".png", img_trans * 255) # 将图像保存到指定文件

        print(img, "done!")
    print('Running time: %s Seconds' % (a)) # 打印去雾总耗时
