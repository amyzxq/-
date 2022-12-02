from django.http import HttpResponse
from web.process import process 
import json
import os
from web.process import getjson

def upload_image(request):
    # 将图片上传到服务器上
    if(request.method == 'POST'):
        print('the POST method')
        path = '/workspace/web/static'
        for i in os.listdir(path):
            path_file = os.path.join(path,i)
            os.remove(path_file)


        img = request.FILES.get("img")
        f = open(os.path.join(path, img.name), 'wb')
        for chunk in img.chunks():
            f.write(chunk)
        f.close()
     
        # 对图片进行分类
        claname = process.classification(img.name) # 图片分类结果

        # 对图片进行解释，首先获取cam图，并保留关键神经元的激活图
        layer, index, weights = process.hcscorecam(img.name)
        
        # 获取关键神经元的类别 
        unitclasses = process.unitclassification(claname, layer, index)
        feature_list = process.relationunit(index, unitclasses, img.name)
        # 将其信息保存在csv中
        message = process.writeCSV(claname, layer, index, weights, unitclasses,feature_list)

        node_labels, node_identity, node_relationship, describe = getjson.promessage(claname, message)
        jsonname = (img.name).split('.')[0] + '.json'
        jsons = getjson.getJson(node_identity, node_relationship, node_labels, describe, jsonname)

        return HttpResponse(json.dumps({"code": 200, "msg": "ok", "data":jsons}))
    return HttpResponse(json.dumps({"code": 400, "msg": "wrong"}))