import torch
import torch.nn.functional as F
import numpy as np
from statistics import mode, mean


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



class ScoreCAM():

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
                # print("predicted class ids {}\t probability {}".format(idx, p))

          
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
            
            for i in range(0, C, self.n_batch):
                mask = self.activations[:, i:i+self.n_batch].transpose(0, 1)  # torch.Size([32, 1, 224, 224])
                mask = mask.to(self.device)
                masked_x = x * mask  # torch.Size([32, 3, 224, 224])
                score = self.model(masked_x) # torch.Size([32, 1000])
                tmp = F.softmax(score, dim=1)[:, idx].to(self.device).data  # torch.Size([32])
                probs.append(tmp)

            
            probs = torch.stack(probs) # concat function
            
            weights = probs.view(1, C, 1, 1).cuda()
            

            # shape = > (1, 1, H, W)
            cam = (weights * self.activations).sum(1, keepdim=True)
            cam = F.relu(cam)
            cam -= torch.min(cam)
            cam /= torch.max(cam)
        return cam.data, weights, self.activations

    def __call__(self, x):
        return self.forward(x)
