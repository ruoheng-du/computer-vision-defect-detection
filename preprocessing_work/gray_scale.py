import numpy as np
import cv2  # https://docs.opencv.org/4.5.5/
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import pywt as wt
import sys, os


# 通过路径获取图像或直接传输图像
def get_img(imgpath, isGray=True):
    try:
        img = cv2.imread(imgpath) ## 导入图片
    except TypeError:
        img = imgpath.copy() ## 复制图片
    else:
        pass

    if isGray:
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## 灰度化：彩色图像转为灰度图像
        return img_g
    else:
        return img


# 绘制图像灰度分布图
def gray_3D(imgpath, newsize):
    img = cv2.imread(imgpath)
    newimg = cv2.resize(img, dsize=None, fx=newsize, fy=newsize, interpolation=cv2.INTER_AREA)
    img_g = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)

    row, col = img_g.shape
    X = np.arange(0, col, step=1)
    Y = np.arange(0, row, step=1)
    xx, yy = np.meshgrid(X, Y)  # 网格化
    X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
    bottom = np.zeros_like(X)  # 设置柱状图的底端位值
    Z = img_g.ravel()
    width = height = 1  # 每一个柱子的长和宽

    # 绘图设置
    fig = plt.figure(1, dpi=300)
    ax = fig.gca(projection='3d')
    ax.bar3d(X, Y, bottom, width, height, Z, shade=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Gray Scale')
    plt.show()

    # cv2.namedWindow('gray',0)
    # cv2.imshow('gray',img_g)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_g


# 增强图像灰度
def gray_enhance(imgpath, ways):
    # https://www.cnblogs.com/supershuai/p/12436669.html#:~:text=OpenCV%E5%9B%BE%E5%83%8F%E5%A2%9E%E5%BC%BA%EF%BC%88python%EF%BC%89.,%E4%B8%BA%E4%BA%86%E5%BE%97%E5%88%B0%E6%9B%B4%E5%8A%A0%E6%B8%85%E6%99%B0%E7%9A%84%E5%9B%BE%E5%83%8F%E6%88%91%E4%BB%AC%E9%9C%80%E8%A6%81%E9%80%9A%E8%BF%87%E6%8A%80%E6%9C%AF%E5%AF%B9%E5%9B%BE%E5%83%8F%E8%BF%9B%E8%A1%8C%E5%A4%84%E7%90%86%EF%BC%8C%E6%AF%94%E5%A6%82%E4%BD%BF%E7%94%A8%E5%AF%B9%E6%AF%94%E5%BA%A6%E5%A2%9E%E5%BC%BA%E7%9A%84%E6%96%B9%E6%B3%95%E6%9D%A5%E5%A4%84%E7%90%86%E5%9B%BE%E5%83%8F%EF%BC%8C%E5%AF%B9%E6%AF%94%E5%BA%A6%E5%A2%9E%E5%BC%BA%E5%B0%B1%E6%98%AF%E5%AF%B9%E5%9B%BE%E5%83%8F%E8%BE%93%E5%87%BA%E7%9A%84%E7%81%B0%E5%BA%A6%E7%BA%A7%E6%94%BE%E5%A4%A7%E5%88%B0%E6%8C%87%E5%AE%9A%E7%9A%84%E7%A8%8B%E5%BA%A6%EF%BC%8C%E8%8E%B7%E5%BE%97%E5%9B%BE%E5%83%8F%E8%B4%A8%E9%87%8F%E7%9A%84%E6%8F%90%E5%8D%87%E3%80%82.%20%E6%9C%AC%E6%96%87%E4%B8%BB%E8%A6%81%E9%80%9A%E8%BF%87%E4%BB%A3%E7%A0%81%E7%9A%84%E6%96%B9%E5%BC%8F%EF%BC%8C%E9%80%9A%E8%BF%87OpenCV%E7%9A%84%E5%86%85%E7%BD%AE%E5%87%BD%E6%95%B0%E5%B0%86%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%E5%88%B0%E6%88%91%E4%BB%AC%E7%90%86%E6%83%B3%E7%9A%84%E7%BB%93%E6%9E%9C%E3%80%82.
    img = get_img(imgpath, False)
    img_g = get_img(imgpath)

    def Normalize():  # 直方图正规化
        Maximg = np.max(img_g)
        Minimg = np.min(img_g)
        Pmin = 0
        Pmax = 255
        a = float((Pmax - Pmin) / (Maximg - Minimg))
        b = Pmin - a * Minimg
        img_ehc = a * img + b
        img_ehc = img_ehc.astype(np.uint8)
        return img_ehc

    def GammaEnhance():
        # 0<gamma<1减弱对比度，gamma>1增强对比度
        gamma = 2.2
        gammaimg = img_g / 255
        img_ehc = np.power(gammaimg, gamma)
        Maximg = np.max(img_ehc)
        img_ehc = img_ehc / Maximg * 255
        img_ehc[img_ehc > 255] = 255
        img_ehc = np.round(img_ehc)
        img_ehc = img_ehc.astype(np.uint8)
        return img_ehc

    def histNoramlize():
        # 直方图的均衡化
        rows, cols = img_g.shape
        grayHist = cv2.calcHist([img_g], [0], None, [256], [0, 255])
        # 计算累加灰度直方图
        zeroCumuMoment = np.zeros([256], np.uint32)
        for p in range(256):
            if p == 0:
                zeroCumuMoment[p] = grayHist[0]
            else:
                zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
        # 根据累加的灰度直方图得到输入与输出灰度级之间的映射关系
        output = np.zeros([256], np.uint8)
        cofficient = 256.0 / (rows * cols)
        for p in range(256):
            q = cofficient * float(zeroCumuMoment[p]) - 1
            if q >= 0:
                output[p] = np.math.floor(q)
            else:
                output[p] = 0
        # 得出均衡化图像
        img_ehc = np.zeros(img_g.shape, np.uint8)
        for r in range(rows):
            for c in range(cols):
                img_ehc[r][c] = output[img_g[r][c]]
        return img_ehc

    if ways != 0 and ways != 1 and ways != 2:
        print('Error: choose the correct method!')
    if ways == 0:
        enhanced = Normalize()
    if ways == 1:
        enhanced = GammaEnhance()
    if ways == 2:
        enhanced = histNoramlize()

    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('enhanced', cv2.WINDOW_NORMAL)
    cv2.imshow('original', img_g)
    cv2.imshow('enhanced', enhanced)
    k = cv2.waitKey(0)
    # 'Esc'退出
    if k == 27:
        cv2.destroyAllWindows()

    return enhanced


# 根据图像灰度值分割图像
def gray_cut(imgpath):
    def TrackBar(a):
        g_min = cv2.getTrackbarPos('gray min', 'TrackBars')
        g_max = cv2.getTrackbarPos('gray max', 'TrackBars')
        print(g_min, g_max)
        return g_min, g_max

    # 创建一个窗口，放置2个滑动条
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 150)
    cv2.createTrackbar('gray min', 'TrackBars', 84, 255, TrackBar)
    cv2.createTrackbar('gray max', 'TrackBars', 255, 255, TrackBar)

    img_g = get_img(imgpath)

    while True:
        img1 = img_g.copy()
        # 调用回调函数，获取滑动条的值
        g_min, g_max = TrackBar(0)
        # 获得指定颜色范围内的掩码
        mask = cv2.inRange(img1, g_min, g_max)
        # 对原图图像进行按位与的操作，掩码区域保留
        imgResult = cv2.bitwise_and(img1, img1, mask=mask)

        cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", imgResult)
        k = cv2.waitKey(50)
        # 'Esc'退出
        if k == 27:
            cv2.destroyAllWindows()
            break

    return g_min, g_max


# 带通滤波器
def bandpass_filter(imgpath: str, radius=200, w=400, n=1, show=True):
    ## https://blog.csdn.net/qq_50587771/article/details/124543040
    ## https://blog.csdn.net/qq_40985985/article/details/119007945?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165147586316782248529605%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165147586316782248529605&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-119007945.142%5Ev9%5Econtrol,157%5Ev4%5Econtrol&utm_term=OpenCV中的傅里叶变换&spm=1018.2226.3001.4187
    '''
    带通滤波函数
    :param image: 输入图像
    :param radius: 带中心到频率平面原点的距离
    :param w: 带宽
    :param n: 阶数
    :param show: 是否展示结果以及调节窗口
    :return: 滤波结果
    '''
    if show == False:
        img_g = get_img(imgpath)
        # 得到中心像素
        rows, cols = img_g.shape[:2]
        mid_row, mid_col = int(rows / 2), int(cols / 2)
        # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
        fft = cv2.dft(np.float32(img_g), flags=cv2.DFT_COMPLEX_OUTPUT)
        # 对fft进行中心化，生成的dshift仍然是一个三维数组
        dshift0 = np.fft.fftshift(fft)
        # 矩形等比掩码，radius为短边宽
        kcr = cols / rows

        dshift = dshift0.copy()
        R = radius
        W = w
        # 构建掩模，256位，两个通道
        mask = np.zeros((rows, cols, 2), np.float32)
        # 矩形等比掩码
        mask[max(0, int(mid_row - R - W)):min(rows, int(mid_row + R + W)),
        max(0, int(mid_col - kcr * (R + W))):min(cols, int(mid_col + kcr * (R + W)))] = 1
        mask[max(0, int(mid_row - R)):min(rows, int(mid_row + R)),
        max(0, int(mid_col - kcr * R)):min(cols, int(mid_col + kcr * R))] = 0

        # 给傅里叶变换结果乘掩模
        fft_filtering = dshift * np.float32(mask)
        # 傅里叶逆变换
        ishift = np.fft.ifftshift(fft_filtering)
        image_filtering = cv2.idft(ishift)
        image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
        # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
        cv2.normalize(image_filtering, image_filtering, 0, 255, cv2.NORM_MINMAX)
        return image_filtering

    def TrackBar(a):
        g_min = cv2.getTrackbarPos('bandpass radius', 'TrackBars')
        g_max = cv2.getTrackbarPos('bandpass width', 'TrackBars')
        print(g_min, g_max)
        return g_min, g_max

    img_g = get_img(imgpath)
    # 得到中心像素
    rows, cols = img_g.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)

    # 创建一个窗口，放置2个滑动条
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 720, 150)
    cv2.createTrackbar('bandpass radius', 'TrackBars', radius, mid_row, TrackBar)
    cv2.createTrackbar('bandpass width', 'TrackBars', w, mid_row, TrackBar)  # 分别调节带通的半径和带宽，半径为0时为低通，带宽最大时为高通

    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv2.dft(np.float32(img_g), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift0 = np.fft.fftshift(fft)
    # 矩形等比掩码，radius为短边宽
    kcr = cols / rows

    while True:
        dshift = dshift0.copy()
        R, W = TrackBar(0)
        ## 带通滤波器源码：https://wenku.baidu.com/view/b5a3558cd2f34693daef5ef7ba0d4a7302766c25.html
        # 构建掩模，256位，两个通道
        mask = np.zeros((rows, cols, 2), np.float32)
        '''    
        # 圆形掩码
        for i in range(0, rows):
            for j in range(0, cols):
                # 计算(i, j)到中心点的距离
                d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
                if radius - W / 2 < d < R + W / 2:
                    mask[i, j, 0] = mask[i, j, 1] = 1
                else:
                    mask[i, j, 0] = mask[i, j, 1] = 0
        '''
        # 矩形等比掩码
        mask[max(0, int(mid_row - R - W)):min(rows, int(mid_row + R + W)),
        max(0, int(mid_col - kcr * (R + W))):min(cols, int(mid_col + kcr * (R + W)))] = 1
        mask[max(0, int(mid_row - R)):min(rows, int(mid_row + R)),
        max(0, int(mid_col - kcr * R)):min(cols, int(mid_col + kcr * R))] = 0

        # 给傅里叶变换结果乘掩模
        fft_filtering = dshift * np.float32(mask)
        # 傅里叶逆变换
        ishift = np.fft.ifftshift(fft_filtering)
        image_filtering = cv2.idft(ishift)
        image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
        # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
        cv2.normalize(image_filtering, image_filtering, 0, 1, cv2.NORM_MINMAX)

        cv2.namedWindow('gray', 0)
        cv2.imshow('gray', img_g)
        cv2.namedWindow('filtering', 0)
        cv2.imshow('filtering', image_filtering)
        k = cv2.waitKey(50)
        # 'Esc'退出
        if k == 27:
            cv2.destroyAllWindows()
            break

    return image_filtering


# SIFT特征识别图像合成
def SIFT_comp(path1, path2, ratio=0.75, reprojThresh=4.0, showMatch=True):
    # https://blog.csdn.net/zhangziju/article/details/79754652
    '''
    ratio: 特征点匹配比率
    reprojThresh: 将点对视为内点的最大允许重投影错误阈值（仅用于RANSAC和RHO方法）
    '''

    # 两图片size保持一致
    img1 = get_img(path1)
    img2 = get_img(path2)
    img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)

    # SIFT特征检测
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    kps1 = np.float32([kp.pt for kp in kp1])
    kps2 = np.float32([kp.pt for kp in kp2])

    ## 全景拼接
    bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    rawMatches = bf.knnMatch(des1, des2, k=2)  # 返回值含义：https://blog.csdn.net/weixin_44072651/article/details/89262277
    # matches = bf.match(des1, des2)
    matches = []  # 用于计算视角变化矩阵，储存匹配对坐标
    goodmatches = []  # 用于绘制匹配点
    for m in rawMatches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # 储存两个点在featuresA， featuresB中的索引值
            matches.append([m[0].trainIdx, m[0].queryIdx])
            goodmatches.append(m[0])

    # 当筛选后的匹配对大于4时， 计算视角变化矩阵
    if matches is None:
        print('无特征对应')
        return None
    if len(matches) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kps1[i] for (_, i) in matches])
        ptsB = np.float32([kps2[i] for (i, _) in matches])

        # 计算视角变化矩阵
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

    # 对图片1进行视角变换
    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    # 图片2传入新图最右端
    tempcol = np.max(np.count_nonzero(result, axis=1)) - 1 ## np.count_nonzero函数介绍：https://blog.csdn.net/zfhsfdhdfajhsr/article/details/109813613
    result[0:img1.shape[0], tempcol:np.min([tempcol + img2.shape[1], result.shape[1]])] = img2[0:img2.shape[0],
                                                                                          0:np.min([img2.shape[1],
                                                                                                    result.shape[
                                                                                                        1] - tempcol])]
    # result=np.hstack((result,img2))
    result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if showMatch:
        # 绘制特征点匹配
        # matches = bf.match(des1, des2)
        # matches = sorted(matches, key=lambda x: x.distance)
        matchresult = cv2.drawMatches(img1, kp1, img2, kp2, goodmatches, None)
        cv2.namedWindow('match', 0)
        cv2.imshow("match", matchresult)
        # cv2.waitKey(0)
    '''
    # SIFT转换结果暂时有问题
    cv2.namedWindow('result', 0)
    cv2.imshow('result', result)
    '''
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result


# 直接调用stitcher类进行图像拼接
def stitcher_comp(path1, path2):
    # 两图片size保持一致
    img1 = get_img(path1, False)
    img2 = get_img(path2, False)
    img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    img = img1, img2
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    (status, pano) = stitcher.stitch(img)  # 不能对灰度图使用
    if status != cv2.Stitcher_OK:
        print("不能拼接图片, error code = %d" % status)  # 失败的话多试几次，可能是每次计算出来的特征点数量不一样，导致不能匹配生成
        sys.exit(-1)
    cv2.namedWindow('result', 0)
    cv2.imshow('result', pano)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pano


# 图像边缘轮廓提取与切割
def contours(imgpath):
    ## https://blog.csdn.net/weixin_44690935/article/details/109008946
    img = get_img(imgpath, False)
    img_g = gray_enhance(imgpath, 1)  # 增强对比度
    ret, binary = cv2.threshold(img_g, 127, 255, cv2.THRESH_BINARY)
    # canny = cv2.Canny(img_g, 50, 200)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 0), 2)
    cv2.namedWindow('contours', cv2.WINDOW_NORMAL)
    cv2.imshow('contours', img)
    cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
    cv2.imshow('origin', img_g)
    k = cv2.waitKey(0)
    # 'Esc'退出
    if k == 27:
        cv2.destroyAllWindows()
    return contours, binary


# 小波变换图像处理
def wavelets_filter(imgpath):
    # https://blog.csdn.net/nanbei2463776506/article/details/64124841
    img_g = get_img(imgpath)
    cA, (cH, cV, cD) = wt.dwt2(img_g, 'dmey')

    iA = cA / np.max(cA) * 255
    iA = iA.astype(np.uint8)
    iH = cH / np.max(cH) * 255
    iH = iH.astype(np.uint8)
    iV = cV / np.max(cV) * 255
    iV = iV.astype(np.uint8)
    iD = cD / np.max(cD) * 255
    iD = iD.astype(np.uint8)

    tempH1 = np.hstack((iA, iH))
    tempH2 = np.hstack((iV, iD))
    rimg = np.vstack((tempH1, tempH2))
    cv2.namedWindow('wavelets', cv2.WINDOW_NORMAL)
    cv2.imshow('wavelets', rimg)
    k = cv2.waitKey(0)
    # 'Esc'退出
    if k == 27:
        cv2.destroyAllWindows()
    return rimg


# Gabor滤波类
class Gabor_filter:
    # https://wangsp.blog.csdn.net/article/details/84841370
    def __init__(self, imgpath=[]):
        self.filter = []
        self.ksize = []
        if imgpath:
            self.img = cv2.imread(imgpath)
            self.img_g = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.img = []
            self.img_g = []

    def getimg(self, path):
        self.img = cv2.imread(path)
        self.img_g = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def build_filter(self, ksize=6, sigma=2.0, theta=4.0, lamda=2, gamma=0.5, psi=0):
        mink = 7
        step = 5
        ksize = range(mink, step * ksize + mink, step)
        self.ksize = ksize
        lamda = np.pi / lamda
        theta = np.arange(0, np.pi, np.pi / theta)
        for i in range(len(theta)):
            for k in range(len(ksize)):
                kern = cv2.getGaborKernel((ksize[k], ksize[k]), sigma, theta[i], lamda, gamma, psi, ktype=cv2.CV_32F)
                kern /= 1.5 * kern.sum()
                self.filter.append(kern)

    def visual_filter(self):
        if not self.filter:
            print('Run function "build_filter" first!')
            return

        fig = plt.figure(dpi=300)
        ax = []
        for i in range(len(self.filter)):
            ax.append(fig.add_subplot(int(len(self.filter) / len(self.ksize)), len(self.ksize), i + 1))
            ax[i].imshow(self.filter[i], cmap='gray')
            ax[i].axis('off')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            # plt.subplot(len(self.filter)/len(self.ksize),len(self.ksize),i+1)
            # plt.imshow(self.filter[i])
        plt.show()
        return fig

    def getGabor(self):
        res = []
        for i in range(len(self.filter)):
            accum = np.zeros_like(self.img_g)
            fimg = cv2.filter2D(self.img_g, cv2.CV_8UC1, self.filter[i])
            accum = np.maximum(accum, fimg)
            res.append(np.asarray(accum))

        fig = plt.figure(dpi=600)
        ax = []
        for i in range(len(res)):
            ax.append(fig.add_subplot(int(len(self.filter) / len(self.ksize)), len(self.ksize), i + 1))
            ax[i].imshow(res[i], cmap='gray')
            ax[i].axis('off')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.show()
        return fig


if __name__ == '__main__':
    path = r"/Users/duruoheng/Desktop/YOLOv5/cylinder/all_images/00-10-21-695_1.jpg"
    # grayimg = gray_3D(path, 0.1)
    # graylim = gray_cut(path)
    img_filter = bandpass_filter(path, 200, 400)
    # img_filter = bandpass_filter(path, 200, 400, show=False)
    # img_SIFT = SIFT_comp(path, path2, ratio=0.75, reprojThresh=6.0)
    # img_stit=stitcher_comp(path,path2)
    # contours, img_contours = contours(path)
    # img_GrayEnhance=gray_enhance(path,2)
    # img_wt = wavelets_filter(path)

    # gabortest = Gabor_filter(path_wenli)
    # gabortest.build_filter(sigma=10.0, theta=4.0, lamda=4, gamma=0.5, psi=0)
    # gabortest.getGabor()

    # graylim=gray_cut(gray_enhance(path,2))

    # n,bins,patches=plt.hist(get_img(path),50)
    # plt.show()
