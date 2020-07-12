"""
利用svd进行图像压缩
"""
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt

def load_data():
    """读取图片并转换为数组，会返回所有通道"""
    pic_path = 'images/xiuqiuhua-001.jpg'
    img = Image.open(pic_path)
    print(np.array(img.getdata()).shape)

    # img.show()
    # data = np.mat(img.getdata())
    data = np.asarray(img)
    print(data.shape)
    print(img.mode)

    return data

def array_to_image(data):
    # img = Image.fromarray(data,mode='RGB')
    # img.save('a.jpg')
    # img.show()
    # cv读取的是BGR，所以换下顺序
    data = data[:,:,::-1]
    # data = cv2.resize(data, (192,120))
    cv2.imwrite('a.jpg',data)


def image_compress(imageData, numSV):
    """
    压缩图片
    :param imageData:(m,n,3)
    :param numSV:奇异值数量
    :param thresh:
    :return:
    """
    # 先将所有通道分开
    channel_num = imageData.shape[2]
    channel_datas = []
    for i in range(channel_num):
        channel_datas.append(np.mat(imageData[:,:,i]))
    # 分别处理每个通道的数据
    channel_datas_recon = []
    for j in range(channel_num):
        data = channel_datas[j]
        U, sigma, VT = np.linalg.svd(data)
        sigRecon = np.mat(np.eye(numSV) * sigma[:numSV])
        recon_data = U[:,:numSV] * sigRecon * VT[:numSV]
        channel_datas_recon.append(recon_data)
    # 合并数据
    channel_datas_recon = np.array(channel_datas_recon)
    new_img_data = np.stack(channel_datas_recon,axis=2)
    return new_img_data





if __name__ == '__main__':
    data = load_data()
    new_data = image_compress(data, 100)
    print(new_data.shape)
    array_to_image(new_data)


