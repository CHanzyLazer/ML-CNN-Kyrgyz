# 独立出来所有的图像处理的方法，除特殊说明都仅可用于灰度图像
import cv2
import numpy as np
import PIL.Image

# PIL型 -> OpenCV型
def pil_to_cv(pil_image: PIL.Image)->np.ndarray:
    cv2_image = np.array(pil_image, dtype=np.uint8)
    if cv2_image.ndim == 2: return cv2_image
    if cv2_image.shape[2] == 3:
        return cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    if cv2_image.shape[2] == 4:
        return cv2.cvtColor(cv2_image, cv2.COLOR_RGBA2BGRA)
    return cv2_image

# 从路径读取图像，解决 opencv 不能读取中文路径的问题
def cv_imread(path: str)->np.ndarray:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

# 裁剪图像的空白区域，可以设置目标裁切颜色和容错率
def cut_around(img: np.ndarray, error=32, around_color=255)->np.ndarray:
    # 使用 where 来避免直接遍历
    x_list, y_list = np.where(np.abs(img.astype(np.int32)-around_color) > error)
    
    if len(x_list) > 0:
        begin_x = np.min(x_list)
        end_x = np.max(x_list)
    else:
        begin_x, end_x = 0, 0
    
    if len(y_list) > 0:
        begin_y = np.min(y_list)
        end_y = np.max(y_list)
    else:
        begin_y, end_y = 0, 0
    
    return img[begin_x:end_x+1, begin_y:end_y+1].copy()

# 将图片周围填充颜色直到其为正方形
def fill_to_square(img: np.ndarray, fill_color=255)->np.ndarray:
    begin_x, begin_y = 0, 0
    end_x, end_y = img.shape
    t_dis = end_x - end_y
    if t_dis == 0:
        return img.copy()
    if t_dis > 0:
        t_size = end_x
        end_y += t_dis//2
        begin_y += t_dis//2
    else:
        t_size = end_y
        t_dis = -t_dis
        end_x += t_dis//2
        begin_x += t_dis//2
    img_out = np.full([t_size, t_size], fill_color, dtype=np.uint8)
    img_out[begin_x:end_x, begin_y:end_y] = img
    return img_out

# 增强图像的对比度到最大
def maximum_contrast(img: np.ndarray)->np.ndarray:
    min_color = np.min(img)
    max_color = np.max(img)
    if (min_color == 0 and max_color == 255) or max_color == min_color:
        return img.copy()
    img_out = np.round((img - min_color) * (255/(max_color-min_color)))
    img_out[img_out > 255] = 255
    img_out[img_out < 0] = 0
    return img_out.astype(np.uint8)

# 将图片周围填充给定大小的包边
def fill_around(img: np.ndarray, length: int, fill_color=255)->np.ndarray:
    img_out = np.full((img.shape[0]+2*length, img.shape[1]+2*length), fill_color, dtype=np.uint8)
    img_out[length:-length, length:-length] = img.copy()
    return img_out

# 获得带有 alpha 通道的图像叠加上纯色背景的图像
def alpha_img_add_bg(alpha_img: np.ndarray, bg_color=(255, 255, 255))->np.ndarray:
    if alpha_img.shape[2] < 4: return alpha_img.copy()
    img_out = np.empty((alpha_img.shape[0], alpha_img.shape[1], 3))
    alpha_layer = np.array(alpha_img[:, :, 3])/255.0
    img_out[:, :, 0] = np.round(alpha_img[:, :, 0] * alpha_layer + bg_color[0] * (1.0-alpha_layer))
    img_out[:, :, 1] = np.round(alpha_img[:, :, 1] * alpha_layer + bg_color[1] * (1.0-alpha_layer))
    img_out[:, :, 2] = np.round(alpha_img[:, :, 2] * alpha_layer + bg_color[2] * (1.0-alpha_layer))
    img_out[img_out > 255] = 255
    img_out[img_out < 0] = 0
    return img_out.astype(np.uint8)
