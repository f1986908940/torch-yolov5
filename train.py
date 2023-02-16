import models
import datasets
import torch 
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class YoloHead(nn.Module):
    
    def __init__(self, num_classes) -> None:
        super().__init__()
        
        self.strides = [8, 16, 32]
        self.anchors = torch.tensor([
            [10,13, 16,30, 33,23],
            [30,61, 62,45, 59,119],
            [116,90, 156,198, 373,326]
        ]).view(3, 3, 2) / torch.FloatTensor(self.strides).view(3, 1, 1)
        self.offset_boundary = torch.FloatTensor([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ])
        self.num_anchor_per_level = self.anchors.size(1)
        self.num_classes = num_classes
        self.anchor_t = 4.0
        self.BCEClassification = nn.BCEWithLogitsLoss(reduce='mean')
        self.BCEObjectness = nn.BCEWithLogitsLoss(reduce='mean')
        self.balance = [4.0, 1.0, 0.4]  # 8 16 32
        
        
        self.box_weight = 0.05
        self.objectness_weight = 1.0
        self.classification_weight = 0.5 * self.num_classes / 80  # 80指coco的类别数
        
        
    def forward(self, predict: Tensor, targets: Tensor):
        '''
        这里计算loss，返回loss
        - predict是模型的预测输出
        - targets是normalize后的标注信息
        '''
        num_target = targets.size(0)
        loss_box_regression = torch.FloatTensor([0])
        loss_classification = torch.FloatTensor([0])
        
        # ilayer = ilevel
        for ilayer, layer in enumerate(predict):
            # b, c, h, w
            layer_height, layer_width = layer.shape[-2:]
            layer = layer.view(-1, self.num_anchor_per_level, 5 + self.num_classes, layer_height, layer_width)
            
            # 转换layer的维度，permute成：batch, num_anchor, height, width, 5+class
            layer = layer.permute(0, 1, 3, 4, 2).contiguous()
            
            # 把targets给转换成基于featuremap大小的框
            # targets[Nx6] -> [image_id, class_index, cx, cy, width, height]
            feature_size_gain = targets.new_tensor([1, 1, layer_width, layer_height, layer_width, layer_height])
            # 形状为Nx6
            targets_feature_scale = targets * feature_size_gain
            
            # 把对应的anchors取出来，形状为3x2
            anchors = self.anchors[ilayer]
            
            # 计算宽宽比，高高比最大值
            anchors_wh = anchors.view(self.num_anchor_per_level, 1, 2)                   # 3x1x2
            targets_wh = targets_feature_scale[:, [4, 5]].view(1, num_target, 2)        # 1xNx2
            
            # shape为 3xNx2
            # 宽宽比，高高比
            wh_ratio = targets_wh / anchors_wh
            
            # 第一个max指anchor / target, target / anchor谁大
            # 第二个max指宽宽比，高高比谁大
            # max_wh_ratio_values形状为 num_anchor x num_target (3xN)
            max_wh_ratio_values, _ = torch.max(wh_ratio, 1 / wh_ratio).max(dim=2)  # 返回值 value, index
            
            # select_mask.dtype = bool
            # select_mask.shape = num_anchor x num_target
            select_mask = max_wh_ratio_values < self.anchor_t
            
            # 把select_mask的所有匹配的target取出来
            # targets_feature_scale.repeat(self.num_anchor_per_level, 1, 1) -> num_anchor x num_target x 6
            # 放了select_mask过后，得到的shape是多少？ -> dim = 2, shape = num_matched x 6
            select_targets = targets_feature_scale.repeat(self.num_anchor_per_level, 1, 1)[select_mask]
            matched_num_target = len(select_targets)
            
            # layer.shape -> [batch, num_anchor, height, width, 5+class]  
            #                                                   [cx, cy, width, height, objectness]
            featuremap_objectness = layer[..., 4]
            objectness_ground_truth = torch.zeros_like(featuremap_objectness)
            
            if matched_num_target > 0: 
                # 如果没有匹配框，不需要做以下计算
                # 这里提取matched_targets对应的anchors
                # anchors.shape -> 3 x 2
                # select_mask.shape -> num_anchor x num_target
                
                # 计算 select_anchors 的索引
                select_anchor_index = torch.range(self.num_anchor_per_level).view(self.num_anchor_per_level, 1).repeat(1, num_target)[select_mask]
                
                # 1.宽宽比，高高比，取最大值，小于阈值anchor_t，被认为是选中   ok
                # 2.拓展样本
                # 3.计算GIoU
                # 4.计算loss
                # 5.loss加权合并
            
                # 先获取到targets的中心点坐标
                # 这里默认就是cx, cy
                # select_targets.shape -> num_matched x 6(image_id, class_index, cx, cy, width, height)
                # select_targets的值域是什么？  是featuremap尺度
                select_targets_xy = select_targets[:, [2, 3]]
                
                # 与1.0取模运算，得到余数  num_matched x 2(cx, cy)
                xy_devided_one_remainder = select_targets_xy % 1.0
                
                # 定义中心位置和上下边界
                coord_cell_middle = 0.5
                feature_map_low_boundary = 1.0
                feature_map_high_boundary = feature_size_gain[[2, 3]] - 1.0
                # less_x_matched.shape  -> num_matched,  dtype = bool
                # less_y_matched.shape  -> num_matched,
                # 左上是否需要增加样本判断
                less_x_matched, less_y_matched = ((xy_devided_one_remainder < coord_cell_middle) & (select_targets_xy > feature_map_low_boundary)).T
                # 右下是否需要增加样本判断
                greater_x_matched, greater_y_matched = ((xy_devided_one_remainder > (1 - coord_cell_middle)) & (select_targets_xy < feature_map_high_boundary)).T
                
                
                # select_anchor_index 相对于原来，增加了多少倍？ 最多增加2倍，大部分时候都是增加2倍
                select_anchor_index = torch.cat([
                    select_anchor_index,
                    select_anchor_index[less_x_matched],      # 左边样本
                    select_anchor_index[less_y_matched],      # 上边样本
                    select_anchor_index[greater_x_matched],   # 右边样本
                    select_anchor_index[greater_y_matched]    # 下边样本
                ], dim=0)

                select_targets = torch.cat([
                    select_targets,
                    select_targets[less_x_matched],      # 左边样本
                    select_targets[less_y_matched],      # 上边样本
                    select_targets[greater_x_matched],   # 右边样本
                    select_targets[greater_y_matched]    # 下边样本
                ], dim=0)
                # 至此完成样本增加
                
                # 计算box的偏移量，xy offset
                # grid_x, grid_y  -> 指featuremap上的坐标
                
                # 给定一个box，计算它的grid_x，grid_y。计算它所属中心位置
                
                # 建立offset，使得增加出来的样本的中心能够计算正确的grid_xy
                xy_offset = torch.zeros_like(select_targets_xy)
                xy_offset = torch.cat([
                    xy_offset,
                    xy_offset[less_x_matched] + self.offset_boundary[0],      # 左边样本
                    xy_offset[less_y_matched] + self.offset_boundary[1],      # 上边样本
                    xy_offset[greater_x_matched] + self.offset_boundary[2],   # 右边样本
                    xy_offset[greater_y_matched] + self.offset_boundary[3]    # 下边样本
                ]) * coord_cell_middle
                
                matched_extend_num_target = len(select_targets)
                gt_image_id, gt_class_id = select_targets[:, [0, 1]].long().T 
                gt_xy = select_targets[:, [2, 3]]
                gt_wh = select_targets[:, [4, 5]]
                grid_xy = (gt_xy - xy_offset).long()
                grid_x, grid_y = grid_xy.T
                
                # 转换为需要回归的xy偏移量
                gt_xy = gt_xy - grid_xy
                
                # 把所有匹配的anchor也提取出来
                select_anchors = anchors[select_anchor_index]
                
                # 开始准备计算GIOU
                # 在这之前，需要把预测框给计算出来
                # layer.shape -> batch, num_anchor, height, width, 5+class
                # 目的：因为要选中predict box，与gt_xy, gt_wh计算它的GIOU，所以需要提取layer中指定项
                # layer中：
                #   - image_id指定的batch
                #   - select_anchor_index指定某个anchor
                #   - grid_x指定width维度
                #   - grid_y指定height维度
                #   - 提取后，得到：num_matched_extend_target x (5 + class)
                # object_predict.shape -> num_matched_extend_target x (5 + class)
                object_predict = layer[gt_image_id, select_anchor_index, grid_y, grid_x]
                # 值域是(-0.5, +1.5)
                object_predict_xy = object_predict[:, [0, 1]].sigmoid() * 2.0 - 0.5
                # 值域是(0, +4)
                object_predict_wh = torch.pow(object_predict[:, [2, 3]].sigmoid() * 2.0, 2) * select_anchors 
                # 拼接为：N x 4，[cx, cy, width, height]
                object_predict_box = torch.cat([object_predict_xy, object_predict_wh], dim=1)
                # 拼接为：N x 4，[cx, cy, width, height]
                object_gt_box = torch.cat([gt_xy, gt_wh], dim=1)
                # 计算GIOU
                gious = self.giou(object_predict_box, object_gt_box)
                giou_loss = 1.0 - gious
                
                loss_box_regression += giou_loss.mean()
                
                # 计算objectness_loss
                # 真值ground_truth是使用GIoU的值作为真值
                # giou的值域是(-1, 1]
                # giou_loss的值域是[0, +2)
                objectness_ground_truth[gt_image_id, select_anchor_index, grid_y, grid_x] = gious.detach().clamp(0)
                
                # 计算类别的loss
                # 如果类别数是一个的话，就直接使用objectness作为目标概率，所以不需要重复计算目标概率
                
                if self.num_classes > 1:
                    # object_classification.shape -> N x num_classes
                    object_classification = object_predict[:, 5:]
                    
                    # 多个二元交叉熵进行分类
                    # 对于检测任务来讲：
                    # 如果有20个类别
                    # predict.shape = height x width x 20
                    # label.shape   = height x width x 20
                    # labelsmooth1 = 如果e = 0.3，类别数是3的时候：0.15 0.7 0.15
                    # labelsmooth2 = 如果e = 0.3，类别数是3的时候：0.1  0.8 0.1
                    classification_targets = torch.zeros_like(object_classification)
                    # gt_class_id.shape -> matched_extend_num_target, 
                    classification_targets[torch.arange(matched_extend_num_target), gt_class_id] = 1.0
                    loss_classification += self.BCEClassification(object_classification, classification_targets)
            
            loss_objectness += self.BCEObjectness(featuremap_objectness, objectness_ground_truth) * self.balance[ilayer]
                
        # 加权求和
        num_level = len(predict)
        scale = 3 / num_level
        
        batch_size = predict[0].shape[0]
        loss_box_regression *= self.box_weight * scale
        loss_objectness *= self.objectness_weight * scale  # 如果 num_level == 4, 这里需要乘以1.4，否则乘以1.0
        loss_classification *= self.classification_weight * scale
        
        loss = loss_box_regression + loss_objectness + loss_classification
        
        return loss * batch_size
                
                
        
    def giou(self, a, b):
        # 计算a和b两坨框的GIoU
        # a[Nx4] cx, cy, width, height
        # b[Nx4] cx, cy, width, height
        
        a_xmin, a_xmax = a[:, 0] - (a[:, 2] - 1) / 2, a[:, 0] + (a[:, 2] - 1) / 2
        a_ymin, a_ymax = a[:, 1] - (a[:, 3] - 1) / 2, a[:, 1] + (a[:, 3] - 1) / 2
        
        b_xmin, b_xmax = b[:, 0] - (b[:, 2] - 1) / 2, b[:, 0] + (b[:, 2] - 1) / 2
        b_ymin, b_ymax = b[:, 1] - (b[:, 3] - 1) / 2, b[:, 1] + (b[:, 3] - 1) / 2

        inter_xmin, inter_xmax = torch.max(a_xmin, b_xmin), torch.min(a_xmax, b_xmax)
        inter_ymin, inter_ymax = torch.max(a_ymin, b_ymin), torch.min(a_ymax, b_ymax)
        
        # 进行裁切，如果小于0，则设置最小值为0，防止宽高都为负数，求面积时导致面积很大
        inter_width = (inter_xmax - inter_xmin + 1).clamp(0)
        inter_height = (inter_ymax - inter_ymin + 1).clamp(0)
        inter_area = inter_width * inter_height
        
        # 计算并集
        union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter_area
        
        # 计算IOU
        iou = inter_area / union
        
        # 最小包裹区域
        convex_width = torch.max(a_xmax, b_xmax) - torch.min(a_xmin, b_xmin) + 1
        convex_height = torch.max(a_ymax, b_ymax) - torch.min(a_ymin, b_ymin) + 1
        convex_area = convex_width * convex_height
        
        # 计算GIOU
        giou = iou - (convex_area - union) / convex_area
        
        return giou
        
        
def train():
    train_set = datasets.VOCDataset(True, 640, '/opt/vscodeprojects/data/VOCdevkit/train/')
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, collate_fn=train_set.collate_fn)

    head = YoloHead(train_set.num_classes)
    model = models.Yolo(train_set.num_classes, '/opt/vscodeprojects/torch-yolov5/models/yolov5s.yaml')
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    for batch_index, (images, targets, visuals) in enumerate(train_loader):
    
        predict = model(images)
        loss = head(predict, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        


if __name__ == '__main__':
    
    train()
