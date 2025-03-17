import glob
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

random.seed(1143)

def populate_train_list(orig_images_path, hazy_images_path):
    train_list, val_list = list(), list()
    tmp_dict = {}
    image_list_haze = glob.glob(hazy_images_path + "*.jpg")
    #glob.glob()返回一个列表，元素为指定hazy_images_path路径下后缀为.jpg的文件路径

    for image in image_list_haze:
        image = image.split("\\")[-1]   # image为hazy_image_path路径下的图片
        #key = image.split("_")[0] + "_" + image.split("_")[1] + '.jpg'
        key = image.split("_")[0] + "_" + image.split("_")[1]
        if key in tmp_dict.keys():
            tmp_dict[key].append(image)
        else:
            tmp_dict[key] = []
            tmp_dict[key].append(image)

    # 构建train_keys、val_keys
    train_keys = []
    val_keys = []
    len_keys = len(tmp_dict.keys()) # hazy_images_path路径下图片的个数
    for i in range(len_keys):
        if i < len_keys:
            train_keys.append(list(tmp_dict.keys())[i])
        else:
            val_keys.append(list(tmp_dict.keys())[i])

    for key in list(tmp_dict.keys()):   # key 图片名称
        if key in train_keys:
            for hazy_image in tmp_dict[key]:
                train_list.append([orig_images_path + key, hazy_images_path + hazy_image])
        else:
            for hazy_image in tmp_dict[key]:
                val_list.append([orig_images_path + key, hazy_images_path + hazy_image])

    random.shuffle(train_list)  # 将列表元素顺序打乱
    random.shuffle(val_list)    # 将列表元素顺序打乱

    return train_list, val_list # 返回train_list、val_list


class dehazing_loader(data.Dataset):
    def __init__(self, orig_images_path, hazy_images_path, mode='train'):
        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path)
        # 确定训练train/验证val？
        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):
        data_orig_path, data_hazy_path = self.data_list[index]
        data_orig = Image.open(data_orig_path)
        data_hazy = Image.open(data_hazy_path)

        data_orig = data_orig.convert("RGB")
        data_hazy = data_hazy.convert("RGB")

        data_orig = data_orig.resize((576, 576), Image.LANCZOS)
        data_hazy = data_hazy.resize((576, 576), Image.LANCZOS)

        data_orig = (np.asarray(data_orig) / 255.0)
        data_hazy = (np.asarray(data_hazy) / 255.0)

        data_orig = torch.from_numpy(data_orig).float()
        data_hazy = torch.from_numpy(data_hazy).float()

        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)   # 交换块和行和列


    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
	orig_images_path = 'train_image/gt/'
	hazy_images_path = 'train_image/hazy/'
	train_loader = dehazing_loader(orig_images_path, hazy_images_path, mode='train')
	for batch_id in range(len(train_loader)):
		x, y = train_loader.__getitem__(batch_id)
		print(x.shape, y.shape)


