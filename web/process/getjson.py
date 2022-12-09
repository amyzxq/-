# 获取前端的json
import os
import json

def getJson(node_identity, node_relationship, node_labels, describe, jsonname):
    """
    前端内容：
    names: ['结果', '权重', '功能', '层数', '神经元'],
    labels: ['Result', 'Weight', 'Function', 'Layer', 'Unit'],
    linkTypes: ['', 'weight', 'function', 'layer','unit']

    node_identity: 节点 以及 identity，字典格式
    node_relationship: 节点 关系 子节点， list格式
    node_labels: 每个节点属于什么类型的标签
    describe: 分类结果的描述
    """ 
    print("try to creat json...")
    path = os.path.join("/workspace/web/src/data", 'result.json')
    if os.path.exists(path):
        os.remove(path)  
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n")
        id = 10000
        for i in range(len(node_relationship)):
            f.write("{\n")
            tmp_list = node_relationship[i].split(":")
            f.write('\t"p":{\n')
            f.write('\t\t"start":{\n')
            f.write('\t\t\t"identity":%s,\n'% node_identity[tmp_list[0]])
            f.write('\t\t\t"labels":["%s"],\n'%node_labels[tmp_list[0]])
            f.write('\t\t\t"properties":{\n')
            if node_labels[tmp_list[0]]=='Result':
                f.write('\t\t\t\t"name":"%s",\n'%tmp_list[0])
                f.write('\t\t\t\t"describe":"%s"}\n'%describe)
            else:
                f.write('\t\t\t\t"name":"%s"}\n'%tmp_list[0])
            f.write("\t\t\t},\n")
            f.write('\t\t"end":{\n')
            f.write('\t\t\t"identity":%s,\n'% node_identity[tmp_list[2]])
            f.write('\t\t\t"labels":["%s"],\n'% node_labels[tmp_list[2]])
            f.write('\t\t\t"properties":{\n')
            f.write('\t\t\t\t"name":"%s"}\n'%tmp_list[2])
            f.write("\t\t\t},\n")
            # 两者之间的关系
            f.write('\t\t"segments":[{\n')
            f.write('\t\t\t"start":{\n')
            f.write('\t\t\t\t"identity":%s,\n'% node_identity[tmp_list[0]])
            f.write('\t\t\t\t"labels":["%s"],\n'% node_labels[tmp_list[0]])
            f.write('\t\t\t\t"properties":{\n')
            if node_labels[tmp_list[0]]=='Result':
                f.write('\t\t\t\t\t"name":"%s",\n'%tmp_list[0])
                f.write('\t\t\t\t\t"describe":"%s"}\n'%describe)
            else:
                f.write('\t\t\t\t\t"name":"%s"}\n'%tmp_list[0])
            f.write("\t\t\t\t},\n")
            f.write('\t\t\t"relationship":{\n')
            f.write('\t\t\t\t"identity":%s,\n'% id)
            id += 10
            f.write('\t\t\t\t"start":%s,\n'% node_identity[tmp_list[0]])
            f.write('\t\t\t\t"end":%s,\n'% node_identity[tmp_list[2]])
            f.write('\t\t\t\t"type":"%s",\n'%tmp_list[1])
            f.write('\t\t\t\t"properties":{\n')
            f.write('\t\t\t\t\t"name":"%s"}\n'%tmp_list[1])
            f.write("\t\t\t\t},\n")
            f.write('\t\t\t"end":{\n')
            f.write('\t\t\t\t"identity":%s,\n'% node_identity[tmp_list[2]])
            f.write('\t\t\t\t"labels":["%s"],\n'%node_labels[tmp_list[2]])
            f.write('\t\t\t\t"properties":{\n')
            f.write('\t\t\t\t\t"name":"%s"}\n'%tmp_list[2])
            f.write("\t\t\t\t}\n")
            f.write('\t\t}],\n')
            f.write('\t\t"length": 1.0\n')
            f.write('\t}\n')
            if i == len(node_relationship) -1: 
                f.write('}\n')
            else: 
                f.write('},\n')
        f.write("]")
        f.close()
        fileJson = open(path)
        fileJson = json.load(fileJson)
        return fileJson

def promessage(claname, message):
    node_identity = {}
    node_labels = {}
    node_relationship = []
    i = 1
    relation = {}
    function = ''
    unit = []
    for tmp in message:
        # <class 'tuple'>
        # ('飞机', '关键神经元', '207')
        # 先将每个节点标上序号，不能重复
        if tmp[0] not in node_identity: 
            node_identity[str(tmp[0])] = i
            i += 1
        if tmp[2] not in node_identity: 
            node_identity[str(tmp[2])] = i
            i += 1

        # 对每个神经元标记自己的类型
        if tmp[0] == '飞机' or tmp[0] == '坦克':
            node_labels[str(tmp[0])] = 'Result'
        else:
            pass
        if tmp[1] == '神经元':
            node_labels[str(tmp[2])] = 'Unit'
        elif tmp[1] == '关键神经元':
            node_labels[str(tmp[2])] = 'impUnit'
            unit.append(tmp[2])
        elif tmp[1] == '层数':
            node_labels[str(tmp[2])] = 'Layer'
        elif tmp[1] == '功能':
            node_labels[str(tmp[2])] = 'Function'
            if str(tmp[2]) == '其他':
                function = '相关背景区域'
            else:
                function = str(tmp[2])
        elif tmp[1] == '权重':
            node_labels[str(tmp[2])] = 'Weight'
            relation[function] = (tmp[2])[0:4]
        elif tmp[1] == '图片':
            node_labels[str(tmp[2])] = 'Unit'
        elif tmp[1] == '相关神经元':
            node_labels[str(tmp[2])] = 'Unit'

        # 生成节点之间对应的关系
        tmp_rela = str(tmp[0])+":"+ str(tmp[1]) +":"+ str(tmp[2])   
        node_relationship.append(tmp_rela)
    
    node_relationship.append(claname+":img:cam.jpg")
    node_identity['cam.jpg'] = i
    node_labels['cam.jpg'] = 'Unit'

    describe = "结果识别为："+claname
    describe = describe + '。原因：从整体情况而言，网络检测出' +claname+'的轮廓，而对于关键语义信息而言，网络检测出图像包含'+str(list(relation.keys()))+'等信息。其中'
    i = 0
    for key in relation:
        describe += str(key) + "主要是在高层语义第"+str(unit[i])+'神经元检测到，权重为'+str(relation[key])+','
        i+=1
    print(relation.keys())
    describe = describe + "综合以上因素，该深度神经元网络模型将该图片识别为" +claname+"。"
    return node_labels, node_identity, node_relationship, describe
# if __name__ == '__main__':
#     node_identity = {"tank": 1, "183":2, "21":3, "56":4, "165":5, "233":6, "254":7, "435":8, "111":9, "478":10, "501":11, "54":12,  "123":13, "174":14, "321":15,
#                      "214": 16, "384":17, "495":18, "247":19, "164":20, "230":21, "0.23":22, "0.31":23, "0.45":24, "0.32":25, "0.21":26, "0.22":27,  "驾驶舱":28, "防浪板":29, "炮筒":30,
#                      "28": 31, "26":32, "28c183.jpg":33, "28c111.jpg":34, "26c123.jpg":35, "26c321.jpg":36,"26c214.jpg":37, "26c495.jpg":38,"26c247.jpg":39,"28c165.jpg":40,"0.34":41,"cam.jpg":42}
#     node_labels = {"tank": 'Result',
#                    "183":'impUnit', "21":'Unit', "56":'Unit', "165":'impUnit', "233":'Unit', "254":'Unit', "435":'Unit', "111":'impUnit', "478":'Unit', "501":'Unit',
#                    "54":'Unit', "123":'impUnit', "174":'Unit', "321":'impUnit', "214":'impUnit', "384":'Unit', "495":'impUnit', "247":'impUnit', "164":'Unit', "230":'Unit',
#                    "0.23":'Weight', "0.34":'Weight',"0.31":'Weight',"0.45":'Weight',"0.32":'Weight',"0.21":'Weight',"0.22":'Weight',"驾驶舱":'Function',"防浪板":'Function',"炮筒":'Function',"28":'Layer',"28c183.jpg":"Unit","28c165.jpg":"Unit","26c247.jpg":"Unit",
#                    "26":'Layer',"28c111.jpg":"Unit","26c123.jpg":"Unit","26c321.jpg":"Unit","26c214.jpg":"Unit","26c495.jpg":"Unit","cam.jpg":"Unit"}
#     node_relationship = ["tank:impunit:183", "tank:unit:21", "tank:unit:56", "tank:impunit:165","tank:unit:233", "tank:unit:254", "tank:unit:435", "tank:impunit:111","tank:unit:478", "tank:unit:501",
#                          "183:impunit:123", "183:impunit:321", "183:impunit:214", "111:impunit:321", "111:impunit:247", "165:impunit:495", "111:impunit:123", "165:impunit:247", "165:impunit:495",
#                          "183:weight:0.23", "183:function:驾驶舱", "183:layer:28","183:img:28c183.jpg",
#                          "165:weight:0.34", "165:function:防浪板", "165:layer:28","165:img:28c165.jpg",
#                          "111:weight:0.31", "111:function:炮筒", "111:layer:28","111:img:28c111.jpg",
#                          "123:weight:0.45",  "123:layer:26","123:img:26c123.jpg",
#                          "321:weight:0.32",  "321:layer:26","321:img:26c321.jpg",
#                          "214:weight:0.21",  "214:layer:26","214:img:26c214.jpg",
#                          "495:weight:0.22",  "495:layer:26","495:img:26c495.jpg",
#                          "247:weight:0.21",  "247:layer:26","247:img:26c247.jpg",
#                          "tank:img:cam.jpg"]
#     describe = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
#     getJson(node_identity, node_relationship, node_labels, describe)