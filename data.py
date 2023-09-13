# 读取所有的图像数据，并使用 numpy 存储来方便训练
import os
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from image import *


class DataKyrgyz:
    def __init__(self, dirname: str):
        # 先确定所有的 label 和文件夹名称
        dir_list =  os.listdir(r"./" + dirname + "/test")
        self.label_list = []
        for t_dir in dir_list:
            t_label = t_dir
            # 乱码处理
            try: t_label = t_label.encode('gbk').decode('utf-8')
            except UnicodeDecodeError: pass
            self.label_list.append(t_label)
        # 将 label 排序
        self.label_list, dir_list = zip(*sorted(zip(self.label_list, dir_list)))
        
        # 根据 dir_list 遍历测试集和训练集，这里断言两者的 dir_list 一致
        self.test_data, self.test_label = self._read_data(dirname + '/test', dir_list)
        self.train_data, self.train_label = self._read_data(dirname + '/train', dir_list)
        
    # 从路径中读取数据，返回数据 ndarray 和标签的 ndarray
    @staticmethod
    def _read_data(path: str, dir_list: list[str])->tuple[np.ndarray, np.ndarray]:
        def read_and_process_image(img_path: str)->np.ndarray:
            return DataKyrgyz.image_preprocess(cv_imread(img_path))
        
        t_data_list = []
        t_label_list = []
        t_label_id = 0
        for t_dir in dir_list:
            t_dir_path = path + '/' + t_dir
            t_filename_list = os.listdir(r"./" + t_dir_path)
            
            t_data_list_dir = Parallel(n_jobs=cpu_count())(delayed(read_and_process_image)(t_dir_path+'/'+t_filename) for t_filename in t_filename_list)
            t_data_list.extend(t_data_list_dir)
            
            t_label_list_dir = np.zeros(len(dir_list), dtype=np.bool_)
            t_label_list_dir[t_label_id] = True # 采用独热编码
            t_label_list.extend([t_label_list_dir]*len(t_filename_list))
            t_label_id += 1
        
        return 1.0 - np.array(t_data_list, dtype=np.float32)/255.0, np.array(t_label_list, dtype=np.bool_)
    
    # 图像预处理，在这里颜色和分辨率不重要，为了降低内存的占用，将三维的输入改为一维并减少分辨率到 32x32
    # 需要将空白区域裁切，然后缩小到 28x28 的大小，然后在周围填充 2 的空白像素
    @staticmethod
    def image_preprocess(img: np.ndarray)->np.ndarray:
        # 为某些有透明部分的添加上白色背景
        img = alpha_img_add_bg(img)
        # 转换为灰度图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 最大化对比度
        img = maximum_contrast(img)
        # 空白像素裁切
        img = cut_around(img)
        # 向周围填充空白保证图像是方的
        img = fill_to_square(img)
        # 缩放 img 到 28x28
        img = cv2.resize(img, (28, 28))
        # 向周围填充白边到 32x32
        img = fill_around(img, 2)
        return img
    
    # 完整的图像处理接口
    @staticmethod
    def image_process(img: np.ndarray) -> np.ndarray:
        img = DataKyrgyz.image_preprocess(img)
        return 1.0 - np.array(img, dtype=np.float32)/255.0
    
    # 提供的实用接口
    def get_label(self, label: np.ndarray)->str: return self.label_list[np.where(label)[0][0]]
    def get_label_list(self)->tuple[str]: return self.label_list
    def get_train_data(self)->tuple[np.ndarray, np.ndarray]: return self.train_data, self.train_label
    def get_test_data(self)->tuple[np.ndarray, np.ndarray]: return self.test_data, self.test_label
    