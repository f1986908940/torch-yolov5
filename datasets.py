import os
import sys
import cv2
import random
import sys_utils
import nn_utils
import numpy as np
from PIL import Image
from sys_utils import _single_instance_logger as logger

import torch
import torchvision.transforms.functional as T


# nn_utils.setup_seed(31)

# 格式定义：
# pixel_annotations:      像素为单位的标注，格式是[left, top, right, bottom]，绝对位置标注
# normalize_annotations:  归一化后的标注0~1，除以图像宽高，格式是[cx, cy, width, height]

class VOCDataset:

    def __init__(self, augment, image_size, root):
        '''
        参数：
            augment: 图像是否增强
            image_size: 图像大小
            root: 图像路径
                train: /opt/vscodeprojects/data/VOCdevkit/train
                test:  /opt/vscodeprojects/data/VOCdevkit/test
        '''
        self.augment = augment
        self.image_size = image_size
        self.root = root
        self.border_fill_value = 114
        self.border_fill_tuple = self.border_fill_value, self.border_fill_value, self.border_fill_value
        self.label_map = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.num_classes = len(self.label_map)
        self.all_labeled_information = []
        
        cache_name = sys_utils.get_md5(root)
        self.build_and_cache(f'runs/dataset_cache/{cache_name}.cache')
           
    def build_and_cache(self, cache_file):
        if os.path.exists(cache_file):
            logger.info(f'Load labels from cache: {cache_file}')
            self.load_labeled_informations_from_cache(cache_file)
        else:
            logger.info(f'Build labels and save to cache: {cache_file}')
            self.build_labeled_information_and_save(cache_file)       
             
    def load_labeled_informations_from_cache(self, cache_file):
        '''
        从缓存文件中加载标注信息
        '''
        self.all_labeled_information = torch.load(cache_file) 
        
    def build_labeled_information_and_save(self, cache_file):
        '''
        主要实现数据检查和缓存数据\n
        缓存的内容为：有效的图像路径，box信息，图像大小
        '''
        annotations_files = os.listdir(os.path.join(self.root, "Annotations"))

        # 保留所有的xml后缀文件
        annotations_files = list(filter(lambda x: x.endswith(".xml"), annotations_files))

        # xml改jpg
        jpeg_files = [item[:-3] + "jpg" for item in annotations_files]
        
        # 把文件名修改为全路径
        annotations_files = map(lambda x: os.path.join(self.root, "Annotations", x), annotations_files)
        jpeg_files = map(lambda x: os.path.join(self.root, "JPEGImages", x), jpeg_files)

        for jpeg_file, annotation_file in zip(jpeg_files, annotations_files):

            # 数据检查
            # 1. 图像是否损坏，如果损坏，直接抛异常
            # 2. 检查图像大小是否过小，如果太小，直接异常
            # 加载标注信息，并保存起来
            #    标注信息是normalize过的

            # 做一个定义
            # 1. 如果基于像素单位的框，绝对位置框，定义为pixel类
            # 2. 如果是归一化后（除以图像宽高的归一化）的框，定义为normalize类
            pil_image = Image.open(jpeg_file)

            # 如果图像不正常，损坏，他会直接给你抛异常
            pil_image.verify()
            
            # 获取exif信息
            image_width, image_height = sys_utils.exif_size(pil_image)
            assert image_width > 9 and image_height > 9, f"Image size is too small{image_width} x {image_height}"
            
            # 加载标注信息，[left, top, right, bottom, class_index]
            pixel_annotations = nn_utils.load_voc_annotation(annotation_file, self.label_map)

            # 转换到normalize，同时变为[cx, cy, width, height]
            normalize_annotations = nn_utils.convert_to_normalize_annotation(pixel_annotations, image_width, image_height)
            self.all_labeled_information.append([jpeg_file, normalize_annotations, [image_width, image_height]])

        sys_utils.mkparents(cache_file)
        torch.save(self.all_labeled_information, cache_file)

    def load_mosaic(self, image_indice):
        '''
        加载图像，并使用马赛克增广
            - 先把1个 image_indice 指定的图，和其他3个随机图拼接为马赛克
            - 使用随机仿射变换，输出image_size大小
            - 移除无效的框
            - 恢复框为normalize \n
        返回值：
            image[self.image_size, self.image_size], normalize_annotations
        '''
        # 在image_size * 0.5到image_size * 1.5之间随机一个中心
        # 马赛克的第一步是拼接为一个大图，即image_size * 2, image_size * 2
        x_center = int(random.uniform(self.image_size * 0.5, self.image_size * 1.5))
        y_center = int(random.uniform(self.image_size * 0.5, self.image_size * 1.5))

        num_images = len(self.all_labeled_information)
        all_image_indices = [image_indice] + [random.randint(0, num_images - 1) for _ in range(3)]

        #  img1,  img2
        #  img3,  img4
        alignment_corner_point = [
            [1, 1],   # img1的角点相对于其宽高尺寸的位置
            [0, 1],   # img2的角点相对于其宽高尺寸的位置
            [1, 0],   # img3的角点相对于其宽高尺寸的位置
            [0, 0]    # img4的角点相对于其宽高尺寸的位置
        ]

        merge_mosaic_image_size = self.image_size * 2
        
        # np.full，如果填充的是tuple，会造成性能严重影响。所以填充的值一定要给int
        merge_mosaic_image = np.full((merge_mosaic_image_size, merge_mosaic_image_size, 3), self.border_fill_value, dtype=np.uint8)
        merge_mosaic_pixel_annotations = []

        for _, (image_indice, (corner_point_x, corner_point_y)) in enumerate(zip(all_image_indices, alignment_corner_point)):

            image, normalize_annotations, (image_width, image_height) = self.load_image_with_uniform_scale(image_indice)

            # 拼接前绘制，用来排查bug
            #nn_utils.draw_norm_bboxes(image, normalize_annotations, color=(0, 0, 255), thickness=3)
            #if index == 0:
            # normalize_annotations = np.zeros((0, 5))

            corner_point_x = corner_point_x * image_width
            corner_point_y = corner_point_y * image_height

            x_offset = x_center - corner_point_x   # 先加目标图的x，再减去image上的x
            y_offset = y_center - corner_point_y

            M = np.array([
                [1, 0, x_offset],
                [0, 1, y_offset]
            ], dtype=np.float32)

            cv2.warpAffine(image, M, (merge_mosaic_image_size, merge_mosaic_image_size), 
                dst=merge_mosaic_image, 
                borderMode=cv2.BORDER_TRANSPARENT,
                flags=cv2.INTER_NEAREST)

            # 把框转换为像素单位，并且是[left, top, right, bottom, class_index]格式的
            pixel_annotations = nn_utils.convert_to_pixel_annotation(normalize_annotations, image_width, image_height)

            # 把框进行平移
            pixel_annotations = pixel_annotations + [x_offset, y_offset, x_offset, y_offset, 0]

            # 把所有框合并
            merge_mosaic_pixel_annotations.append(pixel_annotations)
            
        # 所有框拼接为一个矩阵
        merge_mosaic_pixel_annotations = np.concatenate(merge_mosaic_pixel_annotations, axis=0)

        # 如果框越界了，需要限制到范围内，inplace操作
        np.clip(merge_mosaic_pixel_annotations[:, :4], a_min=0, a_max=merge_mosaic_image_size-1, out=merge_mosaic_pixel_annotations[:, :4])

        # 随机仿射变换
        scale = random.uniform(0.5, 1.5)

        # 随机仿射变化
        #  1. 进行缩放
        #  2. 将large图的中心移动到，目标图(small)的中心上
        #       - large -> 1280 x 1280
        #       - small ->  640 x 640
        # 也知道中心是：self.image_size, self.image_size，缩放系数是scale
        #    x * M00 + 0 * m01 + x_offset    
        #    ix * M00 + x_offset  =  dx
        #       已知一个解，指的是中心点的解
        #       large.center.x * M00 + x_offset = small.center.x
        #       x_offset = small.center.x - large.center.x * M00
        #                = small.center.x - large.center.x * scale
        #                = image_size * 0.5 - image_size * scale
        #                = image_size * (0.5 - scale)
        #   
        #    指定的中心，有什么样的定义：
        M = np.array([
            [scale, 0, self.image_size * (0.5 - scale)],
            [0, scale, self.image_size * (0.5 - scale)]
        ], dtype=np.float32)

        # (B, G, R)
        # (114, 0, 0)
        # warpAffine中提供的borderValue必须是tuple，如果是数值，就等同于(114, 0, 0)，这并不是预期的颜色
        # 而提供tuple，会在c++层面被解析并正确高效的处理，不存在np.full的问题
        # C++   memset  ->   cpu直接提供的指令，是非常高效的
        #  如果提供的是tuple，只能循环赋值，这里指的是np.full
        merge_mosaic_image = cv2.warpAffine(merge_mosaic_image, M, 
            (self.image_size, self.image_size), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_fill_tuple)
        
        # 使用M矩阵对框进行变换，达到目标位置
        num_targets = len(merge_mosaic_pixel_annotations)
        output_normalize_annotations = np.zeros((0, 5))

        if num_targets > 0:
            # 映射标注框到目标图，使用M矩阵
            targets_temp = np.ones((num_targets * 2, 3))

            # 内存排布知识
            # N x 5
            # N x 4 -> left, top, right, bottom, left, top, right, bottom, left, top, right, bottom, 
            #       -> reshape(N x 2, 2)
            #       -> left, top
            #       -> right, bottom
            #       -> left, top
            #       -> right, bottom
            # 把box标注信息变成一行一个点
            targets_temp[:, :2] = merge_mosaic_pixel_annotations[:, :4].reshape(num_targets * 2, 2)

            # targets_temp ->  2N x 3
            # M -> 2 x 3
            # output: 2N x 2,  
            merge_projection_pixel_annotations = merge_mosaic_pixel_annotations.copy()
            merge_projection_pixel_annotations[:, :4] = (targets_temp @ M.T).reshape(num_targets, 4)

            # 处理框
            # 1. 裁切到图像范围
            # 2. 过滤掉无效的框
            np.clip(merge_projection_pixel_annotations[:, :4], a_min=0, a_max=self.image_size-1, out=merge_projection_pixel_annotations[:, :4])

            # 过滤无效的框
            projection_box_width = merge_projection_pixel_annotations[:, 2] - merge_projection_pixel_annotations[:, 0] + 1
            projection_box_height = merge_projection_pixel_annotations[:, 3] - merge_projection_pixel_annotations[:, 1] + 1
            original_box_width = merge_mosaic_pixel_annotations[:, 2] - merge_mosaic_pixel_annotations[:, 0] + 1
            original_box_height = merge_mosaic_pixel_annotations[:, 3] - merge_mosaic_pixel_annotations[:, 1] + 1

            area_projection = projection_box_width * projection_box_height
            area_original = original_box_width * original_box_height

            aspect_ratio = np.maximum(projection_box_width / (projection_box_height + 1e-6), projection_box_height / (projection_box_width + 1e-6))

            # 保留的条件分析
            # 1. 映射后的框，宽度必须大于2
            # 2. 映射后的框，高度必须大于2
            # 3. 裁切后的面积 / 裁切前的面积 > 0.2
            # 4. max(宽高比，高宽比) < 20
            keep_indices = (projection_box_width > 2) & \
                           (projection_box_height > 2) & \
                           (area_projection  / (area_original * scale + 1e-6) > 0.2) & \
                           (aspect_ratio < 20)
            merge_projection_pixel_annotations = merge_projection_pixel_annotations[keep_indices]
            output_normalize_annotations = nn_utils.convert_to_normalize_annotation(merge_projection_pixel_annotations, self.image_size, self.image_size)

        return merge_mosaic_image, output_normalize_annotations
            
    def hsv_augment(self, image, hue_gain=0.015, saturation_gain=0.7, value_gain=0.4):
        '''
        对图像进行HSV颜色空间增广\n
        参数：
            hue_gain:           色调增益，最终增益系数为 random(-1, 1) * hue_gain + 1
            saturation_gain:    饱和度增益，最终增益系数为 random(-1, 1) * saturation_gain + 1
            value_gain:         亮度增益，最终增益系数为 random(-1, 1) * value_gain + 1
        返回值：
            image
        ''' 
        
        # 把增益值修改为最终的增益系数
        hue_gain = np.random.uniform(-1, 1) * hue_gain + 1       
        saturation_gain = np.random.uniform(-1, 1) * saturation_gain + 1       
        value_gain = np.random.uniform(-1, 1) * value_gain + 1
        
        # 把图像转换为HSV之后并分解为H、S、V，3个通道
        hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        
        # cv2.COLOR_BGR2HSV       ->  相对压缩过的，可以说是有点损失的
        # cv2.COLOR_BGR2HSV_FULL  ->  完整的
        
        # hue           -> 值域 0-179
        # saturation    -> 值域 0-255
        # value         -> 值域 0-255
        
        dtype = image.dtype
        lut_base = np.arange(0, 256)
        lut_hue = ((lut_base * hue_gain) % 180).astype(dtype)
        lut_saturation = np.clip(lut_base * saturation_gain, 0, 255).astype(dtype)
        lut_value = np.clip(lut_base * value_gain, 0, 255).astype(dtype)
        
        # cv2.LUT(index, lut)
        changed_hue = cv2.LUT(hue, lut_hue)
        changed_saturation = cv2.LUT(saturation, lut_saturation)
        changed_value = cv2.LUT(value, lut_value)
        
        image_hsv = cv2.merge((changed_hue, changed_saturation, changed_value))
        
        return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
          
    def horizontal_flip(self, image, normalize_annotations):
        '''
        对图像和框进行水平翻转\n
        参数：
            image: 提供图像
            normalize_annotations: 提供归一化后的框信息，格式是[cx, cy, width, height, class_index]
        返回值：
            image, normalize_annotations
        '''
        
        # flipCode = 0      垂直翻转，也就是y轴翻转
        # flipCode = 1      水平翻转，也就是x轴翻转
        # flipCode = -1     对角翻转，x和y都发生翻转
        
        image = cv2.flip(image, flipCode=1)
        
        image_width = image.shape[1]
        normalize_annotations = normalize_annotations.copy()
        normalize_annotations[:, 0] = (image_width - 1) / image_width - normalize_annotations[:, 0]
        return image, normalize_annotations
        
    def load_image_with_uniform_scale(self, image_indice):
        '''
        加载图像并进行长边等比缩放到self.image_size大小
        返回值：
            image, normalize_annotations, (image_resized_width, image_resized_height)
        '''
        jpeg_file, normalize_annotations, (_, _) = self.all_labeled_information[image_indice]
        image = cv2.imread(jpeg_file)

        to_image_size_ratio = self.image_size / max(image.shape[:2])
        if to_image_size_ratio != 1:
            # 如果不需要增广（评估阶段），并且缩放系数小于1，就使用效果比较好的插值方式
            if not self.augment and to_image_size_ratio < 1:
                interp = cv2.INTER_AREA     # 速度慢，效果好，区域插值
            else:
                interp = cv2.INTER_LINEAR   # 速度快，效果也还ok，线性插值

            image = cv2.resize(image, (0, 0), fx=to_image_size_ratio, fy=to_image_size_ratio, interpolation=interp)
        image_resized_height, image_resized_width = image.shape[:2]
        return image, normalize_annotations.copy(), (image_resized_width, image_resized_height)
        
    def load_center_affine(self, image_indice):
        '''
        加载图像，采用中心对齐的方式\n
        返回值:
            image, normalize_annotations
        '''
        image, normalize_annotations, (width, height) = self.load_image_with_uniform_scale(image_indice)
        
        padding_width = self.image_size - width
        padding_height = self.image_size - height
        
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left
        
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        
        image = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left, padding_right, borderType=cv2.BORDER_CONSTANT, value=self.border_fill_tuple)
        
        # 框处理
        # (norm_cx * origin_width + padding_left) / self.image_size    
        #  norm_cx * origin_width / self.image_size + padding_left / self.image_size
        # 令 alpha = origin_width / self.image_size, 
        #    beta = padding_left / self.image_size
        #  
        # norm_width * origin_width / self.image_size
        # norm_width * alpha
        x_alpha = width / self.image_size
        x_beta = padding_left / self.image_size
        
        y_alpha = height / self.image_size
        y_beta = padding_top / self.image_size
        
        # normalize_annotations = normalize_annotations * [x_alpha, y_alpha, x_alpha, y_alpha, 1] + [x_beta, y_beta, 0, 0, 0]
        normalize_annotations[:, [0, 1]] = normalize_annotations[:, [0, 1]] * [x_alpha, y_alpha] + [x_beta, y_beta]
        normalize_annotations[:, [2, 3]] = normalize_annotations[:, [2, 3]] * [x_alpha, y_alpha]
        
        return image, normalize_annotations
    
    
    def __len__(self):
        return len(self.all_labeled_information)
    
    def __getitem__(self, image_idx):
        if self.augment:
            image, normalize_annotations = self.load_mosaic(image_idx)
            
            # 应用HSV增广
            image = self.hsv_augment(image)
            
            # 随机概率进行水平反转
            if random.random() < 0.5:
                image, normalize_annotations = self.horizontal_flip(image, normalize_annotations)
            
            return image, normalize_annotations
        else:
            return self.load_center_affine(image_idx)

    @staticmethod
    def collate_fn(batch):
        '''
        属于dataset.__getitem__之后，dataloader获取数据之前\n
        获取数据之前，指：for batch_index, (images, labels) in enumerate(dataloader):\n
        在这里需要准备一个image_id
        
        因为这里预期返回的内容有：\n
        images[torch.FloatTensor], normalize_annotations[Nx6][image_id, class_index, cx, cy, width, height], visual_info
            - visual_info: 给visdom用来显示的东西。返回batch中某一张图的信息，[image, annotations, image_id]
        '''
        images, normalize_annotations = zip(*batch)
        all_annotations = []
        all_images = []
        
        for index, (annotations, image) in enumerate(zip(normalize_annotations, images)):
            num_targets = len(annotations)
            new_annotations = np.zeros((num_targets, 6))
            new_annotations[:, 0] = index  # image_id
            new_annotations[:, 1:] = annotations[:, [4, 0, 1, 2, 3]]
            all_annotations.append(new_annotations)
            all_images.append(T.to_tensor(image))
            
        # 准备visual_info，用来显示的东西，image, annotations, image_id
        visual_image_id = random.randint(0, len(images) - 1)
        visual_image = images[visual_image_id]
        visual_annotations = normalize_annotations[visual_image_id]
        visual_info = visual_image_id, visual_image, visual_annotations  
        
        all_annotations = torch.FloatTensor(np.concatenate(all_annotations, axis=0))   
        all_images = torch.stack(all_images, dim=0)       

        return all_images, all_annotations, visual_info
    
    

if __name__ == '__main__':

    dataset = VOCDataset(True, 640, "/opt/vscodeprojects/data/VOCdevkit/train/")
    
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=0, collate_fn=dataset.collate_fn)
    
    # for batch in dataloader:
    #     pass
    
    from tqdm import tqdm
    for _ in tqdm(dataset):
        pass
    
    # image, normalize_annotations, (w, h) = dataset.load_image_with_uniform_scale(0)
    # print(image.shape, w, h)
    # print(normalize_annotations)
    
    # nn_utils.setup_seed(31)
    # myimage, _ = dataset.load_mosaic(0)
    
    # pixel_annotations = np.array([
    #     [101.55, 50, 200, 150, 0]
    # ])
    
    # normalize_annotations = dataset.convert_to_normalize_annotations(pixel_annotations, 640, 640)
    # result_pixel_annotations = dataset.convert_to_pixel_annotations(normalize_annotations, 640, 640)
    # print(normalize_annotations.tolist())
    # print(result_pixel_annotations.tolist())
    
    # nn_utils.setup_seed(2)
    # image, normalize_annotations = dataset.load_mosaic(0)
    # print(image.shape)
    # print(normalize_annotations)
    # nn_utils.draw_norm_bboxes(image, normalize_annotations, thickness=2)
    
    # cv2.imwrite('image.jpg', image)
    
    # image, normalize_annotations = dataset.load_center_affine(0)
    # nn_utils.draw_norm_bboxes(image, normalize_annotations, thickness=2)
    # cv2.imwrite('image.jpg', image)
    
    # image, normalize_annotations = dataset.load_center_affine(115)
    # nn_utils.draw_norm_bboxes(image, normalize_annotations, color=(0, 0, 255), thickness=2)
    # image, normalize_annotations = dataset.horizontal_flip(image, normalize_annotations)
    # nn_utils.draw_norm_bboxes(image, normalize_annotations, thickness=2)
    # # image = dataset.hsv_augment(image)
    # cv2.imwrite('image.jpg', image)
    
    
    