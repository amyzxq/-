'''
    计算梯度： 把特征图上对应的某个通道上的每个像素的偏导数求出来后，
              去一次宽高维度上的全局平均，根据其值
              当作该特征图的第k个通道的敏感程度

'''


from statistics import mode
import torch
import os
from torch._C import device
from torchvision import models, transforms
import heapq
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]='5'



normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)


preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

loader = transforms.Compose([
    transforms.ToTensor()])  
    
# 创建hook  直接使用激活值
class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act) # activations values
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad) # gradients values

    def hook_fn_act(self, module, input, output):
        if torch.cuda.is_available():
            self.activations = output.cuda()
        else:
            self.activations = output
  
    def hook_fn_grad(self, module, grad_input, grad_output):
        if torch.cuda.is_available():
            self.gradients = grad_output[0].cuda()
        else:
            self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x


# 梯度计算
class GradCam:
    def __init__(self, model, feature_module, target_layer_names, device):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.device = device
      
        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
 
        features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        # print(index)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)  # 独热编码,shape:(1, 1000)
        one_hot[0,index] = 1  # 独热编码  shape (1, 1000) # one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(self.device) # torch.Size([1, 1000]) # requires_grad_(True)
       
        loss = torch.sum(one_hot * output)
        self.feature_module.zero_grad()  # 将模型的所有参数的梯度清零.
        self.model.zero_grad()  # 将模型的所有参数的梯度清零.
        loss.backward()  # one_hot.backward(retain_graph=True)  


        grads = self.extractor.get_gradients()[-1].cpu().data.numpy()
        return grads


class Score():
    def __init__(self, model, target_layer, x, n_batch=32):
        super().__init__()

        self.device = x.device   
        self.model = model.to(self.device)  # a base model
        self.target_layer = target_layer  # conv_layer you want to visualize 
        self.values = SaveValues(self.target_layer)  # save values of activations and gradients in target_layer
        self.n_batch = n_batch

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: the index of the target class
        Return:
            heatmap: class activation mappings of predicted classes
        """

        with torch.no_grad():
            _, _, H, W = x.shape
            
            score = self.model(x)
            prob = F.softmax(score, dim=1)
            if idx is None:
                p, idx = torch.max(prob, dim=1)
                idx = idx.item()
                print("predicted class ids {}\t probability {}".format(idx, p))


            self.activations = self.values.activations.to(self.device).clone()   # torch.Size([1, 512, 14, 14])
            self.activations = F.relu(self.activations)    # put activation maps through relu activation
            self.activations = F.interpolate(
                self.activations, (H, W), mode='bilinear')  # pytorch 上采样函数
            _, C, _, _ = self.activations.shape

            # normalization
            act_min, _ = self.activations.view(1, C, -1).min(dim=2)
            act_min = act_min.view(1, C, 1, 1)
            act_max, _ = self.activations.view(1, C, -1).max(dim=2)
            act_max = act_max.view(1, C, 1, 1)
            denominator = torch.where(
                (act_max - act_min) != 0., act_max - act_min, torch.tensor(1.).to(self.device)
            )
            self.activations = self.activations / denominator    # torch.Size([1, 512, 224, 224])
            
          
            # generate masked images and calculate class probabilities
            probs = []
            maskeds = []
            for i in range(0, C, self.n_batch):
                mask = self.activations[:, i:i+self.n_batch].transpose(0, 1)  # torch.Size([32, 1, 224, 224])
                mask = mask.to(self.device)
                masked_x = x * mask  # torch.Size([32, 3, 224, 224])
                maskeds.append(mask)
                score = self.model(masked_x) # torch.Size([32, 1000])
                tmp = F.softmax(score, dim=1)[:, idx].to(self.device).data  # torch.Size([32])

                probs.append(tmp)
               
            
            probs = torch.stack(probs) # concat function 
            weights = probs.view(1, C, 1, 1).cuda()

       

            
        return weights

    def __call__(self, x):
        return self.forward(x)


def rot180(conv_filters):
    rot180_filters = np.zeros((conv_filters.shape))
    for filter_num in range(conv_filters.shape[0]):
            rot180_filters[filter_num,:,:] = np.flipud(np.fliplr(conv_filters[filter_num,:,:]))
    return rot180_filters


def backpro(x, weight, bias=0, stride=1, pad=1): 

    h_in, w_in = x.shape
    x_pad = torch.zeros(h_in+2*pad, w_in+2*pad)   # 对输入进行补零操作
    if pad>0:
        x_pad[pad:-pad, pad:-pad] = torch.from_numpy(x)
    else:
        x_pad = x
    # print(x_pad)
    h_in, w_in = x_pad.shape
    n, w_height, w_width = weight.shape
    conv_heigh = h_in - w_height + 1
    conv_width = w_in - w_width + 1
    conv = np.zeros((conv_heigh,conv_width),dtype = 'float32')
    backpro_conv = np.zeros((512, conv_heigh, conv_width), dtype=np.float32)
    for i in range(n):
        fil = weight[i] # 单独一个kernel
        # print(fil)
        for c_h in range(conv_heigh):
                for c_w in range(conv_width):
                    conv[c_h][c_w] = (x_pad[c_h:c_h + w_height,c_w:c_w + w_width ] * fil).sum()  
        # print(conv)
        backpro_conv[i] = conv
    return backpro_conv


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    picture = Image.open(r'/root/jzb/shiyan2/airplane_8.jpg')
    tensor = preprocess(picture)
    tensor = tensor.unsqueeze(0)
    if torch.cuda.is_available(): 
        tensor = tensor.cuda()
    
    
    model = models.vgg16(pretrained=True).to(device)
    model.eval()
    channels = (model.features)[28].weight.size()[0] # 通道数


    # score 置信度计算
    target_layer = (model.features)[28]
    wrapped_model = Score(model, target_layer, tensor)
    score = wrapped_model(tensor)  
    score = score.squeeze() 
    max_num_index_score=heapq.nlargest(10, range(len(score)), score.__getitem__)
    print(max_num_index_score)
    for i in max_num_index_score:
        print(i)
        print(score[i])
  
    # # # 先计算出最后一层卷积的梯度---第28层  （梯度计算使用梯度久三）
    grad_cam = GradCam(model=model, feature_module=model.features, \
                    target_layer_names=[str(28)], device=device)
  
    grads_28 = grad_cam(tensor, index=None) # (1, 512, 14, 14)
    

    for i in max_num_index_score:
        # 再计算出第26层的梯度 
        grads_28_tmp = grads_28[0, i] # 取出单个某个通道 (14, 14)      
        for name, parameters in model.named_parameters():
            if name == 'features.28.weight':
                output, input, n, _ = parameters.size() # [512, 512, 3, 3]
                tmp_para = parameters[i].cpu().data.numpy() # 得到对应通道的卷积参数
                tmp_para_rot180 = rot180(tmp_para) # rot180°卷积 (512, 3, 3)
                backpro_conv = backpro(x=grads_28_tmp, weight=tmp_para_rot180) # (512, 14, 14)
                sum_tmp = backpro_conv.sum(axis=(1,2))
                max_num_index_grad = heapq.nlargest(10, range(len(sum_tmp)), sum_tmp.__getitem__)
                print(i)
                print(max_num_index_grad)
          



