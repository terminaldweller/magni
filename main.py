#!/usr/bin/env python
"""Superscaler."""
# https://learnopencv.com/super-resolution-in-opencv/
# https://github.com/xinntao/ESRGAN
# https://github.com/xinntao/Real-ESRGAN
# https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan

# modls
# https://github.com/Saafke/EDSR_Tensorflow/tree/master/models
# https://github.com/fannymonori/TF-ESPCN/tree/master/export
# https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models
# https://github.com/fannymonori/TF-LapSRN/tree/master/export

import cv2
import time


def edsr_superscaler(img):
    """EDSR superscaler."""
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "./models/EDSR_x4.pb"
    sr.readModel(path)
    sr.setModel("edsr", 4)
    result = sr.upsample(img)
    return result


def espcn_superscaler(img):
    """ESPCN superscaler."""
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "./models/ESPCN_x3.pb"
    sr.readModel(path)
    sr.setModel("espcn", 3)
    result = sr.upsample(img)
    return result


def fsrcnn_superscaler(img):
    """FSRCNN superscaler"""
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "./models/FSRCNN_x3.pb"
    sr.readModel(path)
    sr.setModel("fsrcnn", 3)
    result = sr.upsample(img)
    return result


def lapsrn_superscaler(img):
    """LapSRN superscaler"""
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "./models/LapSRN_x8.pb"
    sr.readModel(path)
    sr.setModel("lapsrn", 8)
    result = sr.upsample(img)
    return result


def main():
    img = cv2.imread("./21-o.jpg")

    time_begin = time.time()
    edsr_super_img = edsr_superscaler(img)
    cv2.imwrite("./edsr_super.jpg", edsr_super_img)
    time_end = time.time()
    print("time {}".format(time_end - time_begin))

    time_begin = time.time()
    espcn_super_img = espcn_superscaler(img)
    cv2.imwrite("./espcn_super.jpg", espcn_super_img)
    time_end = time.time()
    print("time {}".format(time_end - time_begin))

    time_begin = time.time()
    fsrcnn_super_img = fsrcnn_superscaler(img)
    cv2.imwrite("./fsrcnn_super.jpg", fsrcnn_super_img)
    time_end = time.time()
    print("time {}".format(time_end - time_begin))

    time_begin = time.time()
    lapsrn_super_img = lapsrn_superscaler(img)
    cv2.imwrite("./lapsrn_super_img.jpg", lapsrn_super_img)
    time_end = time.time()
    print("time {}".format(time_end - time_begin))


if __name__ == "__main__":
    main()
