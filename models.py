import torch
import torch.nn as nn
import yaml
import math
import nn_utils

def autopadding(kernel_size, padding=None):
    # Pad to 'same'
    if padding is None:
        padding = kernel_size // 2
    return padding


class Conv(nn.Module):
    '''
    标准卷积层
    '''
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, groups=1, activation=True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=autopadding(kernel_size, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(0.1, inplace=True) if activation else nn.Identity()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    
class Bottleneck(nn.Module):
    '''
    标准瓶颈层
    '''
    
    def __init__(self, in_channel, out_channel, shortcut=True, groups=1, expansion=0.5) -> None:
        super().__init__()
        
        hidden_channel = int(out_channel * expansion)
        
        self.conv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.conv2 = Conv(hidden_channel, out_channel, 3, 1, groups=groups)
        self.add = shortcut and in_channel == out_channel
        
    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        return (x + x_out) if self.add else x_out
    

class BottleneckCSP(nn.Module):
   
    def __init__(self, in_channel, out_channel, repeats=1, shortcut=True, groups=1, expansion=0.5) -> None:
        super().__init__()
        hidden_channel = int(out_channel * expansion)
        
        self.conv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.conv2 = nn.Conv2d(in_channel, hidden_channel, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hidden_channel, hidden_channel, 1, 1, bias=False)
        self.conv4 = Conv(2 * hidden_channel, out_channel, 1, 1)
        self.bn = nn.BatchNorm2d(hidden_channel * 2)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
        self.repeat_blocks = nn.Sequential(*[
            Bottleneck(hidden_channel, hidden_channel, shortcut, groups, expansion) for _ in range(repeats)
        ])
        
    def forward(self, x):
        y1 = self.conv3(self.repeat_blocks(self.conv1(x)))
        y2 = self.conv2(x)
        ycat = torch.cat((y1, y2), dim=1)
        return self.conv4(self.act(self.bn(ycat)))
    

class SPP(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel_size_list=(5, 9 ,13)) -> None:
        super().__init__()
        hidden_channel = in_channel // 2
        
        self.conv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.conv2 = Conv(hidden_channel * (len(kernel_size_list) + 1), out_channel, 1, 1)
        
        self.spatial_pyramid_poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2) for kernel_size in kernel_size_list
        ])
        
        
    def forward(self, x):
        x = self.conv1(x)
        spp = torch.cat([x] + [m(x) for m in self.spatial_pyramid_poolings], dim=1)
        return self.conv2(spp)
        
        
class Focus(nn.Module):
    '''
    一种下采样方式，通过切片的形式将输入进行下采样2倍，但是不损失信息
    input[1x3x100x100]
    output[1x12x50x50]
    '''
    
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, activation=True) -> None:
        super().__init__()
        self.conv = Conv(in_channel * 4, out_channel, kernel_size, stride, padding, groups, activation)
        
    def forward(self, x):
        # block(y, x)
        # a(0, 0)   b(1, 0)
        # c(0, 1)   d(1, 1)
        a = x[..., ::2, ::2]
        b = x[..., 1::2, ::2]
        c = x[..., ::2, 1::2]
        d = x[..., 1::2, 1::2]
        return self.conv(torch.cat([a, b, c, d], dim=1))
    

class Concat(nn.Module):
    
    def __init__(self, dimension=1) -> None:
        super().__init__()
        self.dimension = dimension
        
    def forward(self, x):
        return torch.cat(x, dim=self.dimension)
    

class Detect(nn.Module):
    
    def __init__(self, num_class, num_anchor, reference_channels) -> None:
        super().__init__()
        
        self.num_anchor = num_anchor
        self.num_classes = num_class
        self.num_output = self.num_classes + 5
        
        self.heads = nn.ModuleList([
            nn.Conv2d(input_channel, self.num_anchor * self.num_output, 1) for input_channel in reference_channels
        ])
        
    def forward(self, x):
        for ilevel, head in enumerate(self.heads):
            x[ilevel] = head(x[ilevel])
        return x
    
    
class Yolo(nn.Module):
    
    def __init__(self, num_classes, config_file) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.model, self.saved_index, self.anchors = self.build_model(config_file)
        
    def forward(self, x):
        y = []
        for module_instance in self.model:
            if module_instance.from_index != -1:
                if isinstance(module_instance.from_index, int):
                    x = y[module_instance.from_index]
                else:
                    xout = []
                    for i in module_instance.from_index:
                        if i == -1:
                            xvalue = x
                        else:
                            xvalue = y[i]
                        xout.append(xvalue)
                    x = xout
            x = module_instance(x)
            if module_instance.layer_index in self.saved_index:
                y.append(x)
            else:
                y.append(None)
        return x
                
        
        
    def parse_string(self, value):
        if value == 'None':
            return None
        elif value == 'True':
            return True
        elif value == 'False':
            return False
        else:
            return value
        
    def make_divisible(self, value, divisor):
        return math.ceil(value / divisor) * divisor
        
    def build_model(self, config_file, input_channel=3):
        with open(config_file) as f:
            self.yaml = yaml.load(f, Loader=yaml.FullLoader)
        
        # list[from, repeat, module_name, args]
        layers_cfg_list = self.yaml['backbone'] + self.yaml['head']
        
        anchors, depth_multiple, width_multiple = [self.yaml[item] for item in ['anchors', 'depth_multiple', 'width_multiple']]
        
        # 每个level所具有的anchor数量，这里是3
        num_anchor_per_level = len(anchors[0]) // 2
        
        # [cx, cy, w, h, objectness, num_classes * class_prob] * num_anchor_per_level
        num_output_per_level = (5 + self.num_classes) * num_anchor_per_level
        
        # 用来存储所有layer的输出通道数
        layers_channel = [input_channel]
        
        # 用来存储所有的layer
        layers = []
        
        # 用来使得推理时能够根据from_index获取到对应的layer输出
        # 其实也是告诉推理过程，你需要保留哪一些layer的输出
        saved_layer_index = []
        
        for layer_index, (from_index, repeat, module_name, args) in enumerate(layers_cfg_list):
            args = [self.parse_string(item) for item in args]
            # 这里是class的引用，不是实例
            module_class_reference = eval(module_name)
            
            if repeat > 1:
                repeat = max(round(repeat * depth_multiple), 1)
            
            if module_class_reference in [Conv, Bottleneck, BottleneckCSP, SPP, Focus]:
                
                input_channel = layers_channel[from_index]
                output_channel = args[0]
                
                if output_channel != num_output_per_level:
                    output_channel = self.make_divisible(output_channel * width_multiple, 8)
                    
                # args[0]是ouput_channel，那么1：参数，是不是就属于layer的特定参数
                args = [input_channel, output_channel, *args[1:]]
                if module_class_reference in [BottleneckCSP]:
                    args.insert(2, repeat)
                    repeat = 1
            elif module_class_reference is Concat:
                output_channel = 0
                for index in from_index:
                    if index != -1:
                        index += 1
                    output_channel += layers_channel[index]
            elif module_class_reference is Detect:
                reference_channel = [layers_channel[index + 1] for index in from_index]
                args = [self.num_classes, num_anchor_per_level, reference_channel]
            else:
                output_channel = layers_channel[from_index]
                
            layers_channel.append(output_channel)
            
            # 开始基于repeat构建重复模块
            if repeat > 1:
                module_instance = nn.ModuleList([
                    module_class_reference(*args) for _ in range(repeat)
                ])
            else:
                module_instance = module_class_reference(*args)
                
            module_instance.from_index = from_index
            module_instance.layer_index = layer_index
            layers.append(module_instance)
            
            if not isinstance(from_index, list):
                from_index = [from_index]
                
            saved_layer_index.extend(filter(lambda x: x != -1, from_index))
            
        return nn.Sequential(*layers), sorted(saved_layer_index), anchors


    
if __name__ == '__main__':
    # spp = SPP(64, 128)
    # x = torch.randn((4, 64, 256, 256))
    # print(x.shape)
    
    # out = spp(x)
    # print(out.shape)
    # detect = Detect(20, 3, (128,))
    # pred = detect(out)
    # print(pred.shape)
    
    nn_utils.setup_seed(3)
    model = Yolo(20, '/opt/vscodeprojects/torch-yolov5/models/yolov5m.yaml')
    input = torch.zeros((1, 3, 64 ,64))
    y = model(input)
    
    print(y[0][0, 0, 0, 0].item())