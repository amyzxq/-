#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from torchvision.utils import save_image
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F
from .hcscorecam import ScoreCAM
import torch.nn as nn
from .visualize import visualize, reverse_normalize
import os, csv, codecs, torch, heapq
from web.process import vgg16
from .fanxiang import GradCam, rot180, backpro
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def classification(picname):
    '''
        对上传的图片进行分类

        picnamde: 图片名称

        return: 
            图片所属类别
    '''
    dirname = '/workspace/web/static'  # 图片保存的路径
    picturepath = os.path.join(dirname, picname)
    picture = Image.open(picturepath)
    tensor = preprocess(picture)
    tensor = tensor.unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    # 分类
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg = vgg16.VGG16(num_classes=2).to(device)
    vgg.load_state_dict(torch.load('/workspace/web/model/picmodel.pt'))    # 加载模型
    vgg.eval()  
    score = vgg(tensor)
    pred = score.argmax(dim=1)
    print(score)
    return '飞机' if pred==0 else '坦克'

def unitclassification(picclass, layer, index):
    '''
    对重要神经元进行分类
    picclass:图片所属类别
    layer:神经元所在层数
    index:神经元序号
    
    '''
    labels = {0:"机身", 1:"机头", 2:"其他", 3:"机尾", 4:"机翼", 5:"轮胎", 6:"防浪板", 7:"驾驶舱", 8:"履带", 9:"炮筒", 10:"天线"}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg = vgg16.VGG16(num_classes=11).to(device)
    vgg.load_state_dict(torch.load('/root/jzb/web/model/pre_model.pt'))  
    # vgg = torch.load('/root/jzb/shiyan2/data/model.pkl')  #加载模型
    vgg.eval()  
    unitclasses = []
    for i in index:
        unitpath = os.path.join('/root/jzb/web/static',str(layer)+'C'+str(i)+'.jpg')
        picture = Image.open(unitpath)
        tensor = preprocess(picture)
        tensor = tensor.unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        outputs = vgg(tensor)
        # pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        outputs = outputs.to('cpu')
        outputs = outputs[0].detach().numpy()
        if picclass == '飞机':
            pred = np.argmax(outputs[0:6])
            claname = labels[pred]
        else:
            pred = np.argmax(outputs[6:]) + 6
            pred = 2 if outputs[pred] < outputs[2] else pred
            claname = labels[pred]
        unitclasses.append(claname)
    return unitclasses

def hcscorecam(picname):
    '''
    获得hcscorecam图

    picname:图片名称

    return:
        hcscorecam图
    '''
    dirname = '/root/jzb/web/static'
    picturepath = os.path.join(dirname, picname)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score_saliency_map = torch.zeros((1, 1, 224, 224))
    model = models.vgg16(pretrained=True).to(device)
    model.eval()
    layers = [5, 24, 26,28]
    for i in layers:
        target_layer = (model.features)[i]
        image = Image.open(picturepath)
        tensor = preprocess(image)
        tensor = tensor.unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()


        wrapped_model = ScoreCAM(model, target_layer, tensor)

        cam, weights, activations = wrapped_model(tensor)   # 类激活图，权重，激活图
        cam = cam.to('cpu')
        score_saliency_map += cam
         
    score_saliency_map = F.relu(score_saliency_map)
    score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()
    score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data
    img = reverse_normalize(tensor)
    visualize(img, score_saliency_map)

    # 置信度
    cic_score = weights.squeeze() 

    # 反向传播
    grad= GradCam(model=model, feature_module=model.features, \
                        target_layer_names=[str(28)], device=device)
    
    score_grad = grad(tensor, index=None) # (1, 512, 14, 14)

    score_grad = np.mean(score_grad, axis=(2, 3))
    score_grad = torch.from_numpy(score_grad)
    score_grad = score_grad.squeeze() 
    
    # 重要神经元
    score = score_grad * cic_score
    max_num_index_score=heapq.nlargest(10, range(len(score)), score.__getitem__)
    activations = activations.to('cpu')
    weights = []
    picture = loader(image)
    for j in max_num_index_score:
        heatmap = activations[0,j]
        weights.append(score[j])
        heatmap=heatmap.detach().numpy()
        max_pixel = np.max(heatmap)
        np.putmask(heatmap, heatmap <= max_pixel * 0.55, 0.)
        heatmap = torch.from_numpy(heatmap)
        image = torch.mul(picture, heatmap)
        save_image(image, os.path.join('/root/jzb/web/static',str(i)+'C'+str(j)+'.jpg'))

    return layers[-1], max_num_index_score, weights


    # score = weights.squeeze() 
    # max_num_index_score=heapq.nlargest(10, range(len(score)), score.__getitem__)
    # activations = activations.to('cpu')
    # weights = []
    # picture = loader(image)
    # for j in max_num_index_score:
    #     heatmap = activations[0,j]
    #     weights.append(score[j])
    #     heatmap=heatmap.detach().numpy()
    #     max_pixel = np.max(heatmap)
    #     np.putmask(heatmap, heatmap <= max_pixel * 0.55, 0.)
    #     heatmap = torch.from_numpy(heatmap)
    #     image = torch.mul(picture, heatmap)
    #     save_image(image, os.path.join('/root/jzb/web/static',str(i)+'C'+str(j)+'.jpg'))

    # return layers[-1], max_num_index_score, weights

def relationunit(index, unitclasses, picname):
    '''
    对关键神经元进行反向传播，获得相连层的相关神经元

    index:神经元序号
    unitclasses:关键神经元类别
    picname:图片名称

    return :
            相关神经元
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(pretrained=True).to(device)
    model.eval()
    weights = []
    for name,parameters in model.named_parameters():
        if ('feature' in name) and ('weight' in name):
            weights.append(parameters)
    dirname = '/workspace/web/static'
    picturepath = os.path.join(dirname, picname)
    picture = Image.open(picturepath)
    tensor = preprocess(picture)
    tensor = tensor.unsqueeze(0)
    if torch.cuda.is_available(): 
        x = tensor.cuda()
    grad_cam = GradCam(model=model, feature_module=model.features, \
            target_layer_names=[str(26)], device=device)
    grads = grad_cam(x, index=None) # (1, 512, 14, 14)    
    flag = [] 
    feature_list = []
    for i in range(len(index)):
        # print(index[i])
        if unitclasses[i] not in flag:
    # 确定之前关键的神经元
    # 上一层的梯度层的梯度
            grads_tmp = grads[0, index[i]]
            weight = nn.Parameter(weights[-1]).cuda()  # 自定义的权值
            tmp_para = weight[index[i]].cpu().data.numpy() # 得到对应通道的卷积参数 [512, 512, 3, 3]
            tmp_para_rot180 = rot180(tmp_para) # rot180°卷积 (512, 3, 3)
            backpro_conv = backpro(x=grads_tmp, weight=tmp_para_rot180) # (512, 14, 14)
            sum_tmp = backpro_conv.sum(axis=(1,2))
            max_num_index_grad = heapq.nlargest(2, range(len(sum_tmp)), sum_tmp.__getitem__)
            print(index[i])
            print(max_num_index_grad)
            feature_list.append(max_num_index_grad)
            flag.append(unitclasses[i])
    return  feature_list

def writeCSV(claname, layer, index, weights, unitclasses, feature_list):
    '''
        将获取到的信息保存在csv文件中

        claname:图片类别
        layer:所在层
        index:序号
        weights:权重
        unitclasses:神经元所属类别（语义信息）
        feature_list: 对应相关神经元

        return:
            csv信息

    '''
    air = ['6','51','93','105','112','137','143','176','207','289']
    tank = ['127','128','165','183','369','416','448','472','490','495']
    flag = [] 
    message = []
    # tmp = [['213','43'],['67','341'],['76','56'],['126','432'],['41','433'],['314','286'],['23','4'],['465','422']]
    j = 0
    if claname == '飞机':
        for i in air: message.append((claname, '神经元', str(i)))
    else:
        for i in tank: message.append((claname, '神经元', str(i)))
    for i in range(len(index)):
        # print(index[i])
        if unitclasses[i] not in flag:
            message.append((claname, '关键神经元', str(index[i])))
            message.append((str(index[i]), '层数', str(layer)))
            message.append((str(index[i]), '功能', str(unitclasses[i])))
            message.append((str(index[i]), '权重', str(weights[i].item())))
            message.append((str(index[i]), '图片', str(layer)+'C'+str(index[i])+'.jpg'))
            for k in feature_list[j]:
                message.append((str(index[i]), '相关神经元', str(k)))
            j += 1    
            flag.append(unitclasses[i])

    f = codecs.open('/workspace/web/model/result.csv','w','gbk')
    writer = csv.writer(f)
    for i in message:
        writer.writerow(i)
    f.close()
    return message

if __name__ == '__main__':
    classification('tank_1.jpg')


