'''
分类移动图片位置
'''

import os, re, shutil


def Classify(path,dst):
    os.chdir(path) ## 修改文件路径
    filelist = os.listdir(path) ## 返回文件列表（字母顺序）
    part = r'_1'
    part = re.compile(part) ## re.compile()介绍：https://blog.csdn.net/weixin_39662578/article/details/113499179
    for i in range(len(filelist)):
        res = re.search(part, filelist[i])
        if res != None:
            ## shutil介绍：https://baijiahao.baidu.com/s?id=1716363868234453618&wfr=spider&for=pc
            shutil.copy(os.path.join(path, filelist[i]), dst) ## 拷贝每个文件已指定名字进入目标文件夹
            # print(filelist[i])
    return filelist


if __name__ == '__main__':
    path = r'/Users/duruoheng/Desktop/YOLOv5/cylinder/all_images' # 图片所在源文件夹
    dstpath = r'/Users/duruoheng/Desktop/YOLOv5/cylinder/round1_images' # 目标文件夹，需要先创建文件夹
    imglist = Classify(path, dstpath)
