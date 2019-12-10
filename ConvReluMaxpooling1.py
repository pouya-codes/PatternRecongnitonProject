# Implimentation of Conv, ReLU and MaxPooling functions
import skimage.data
import numpy as np
import matplotlib
import sys
import cv2
import matplotlib.pyplot as plt

def myconv_(input, kernel):
    kernel_d = kernel.shape[1]
    outputs = np.zeros((input.shape))
    h,w = input.shape
    a1 = (np.arange(kernel_d/ 2.0, h - kernel_d/ 2.0 + 1)).astype(int)
    a2 = (np.arange(kernel_d/ 2.0, w - kernel_d/ 2.0 + 1)).astype(int)
    for x in a1:
        for y in a2:
            part_res = (input[x - np.uint8(np.floor(kernel_d/ 2.0)):x + np.uint8(np.ceil(kernel_d/ 2.0)),
                          y - np.uint8(np.floor(kernel_d/ 2.0)):y + np.uint8(np.ceil(kernel_d/ 2.0))]) * kernel
            part_sum = np.sum(part_res)
            outputs[x, y] = part_sum

    myconv_out = outputs[np.uint8(kernel_d / 2.0):outputs.shape[0] - np.uint8(kernel_d / 2.0),
                   np.uint8(kernel_d / 2.0):outputs.shape[1] - np.uint8(kernel_d / 2.0)]
    return myconv_out


def conv_func(input, kernel):
    conv_func_output = np.zeros((input.shape[0] - kernel.shape[1] + 1, input.shape[1] - kernel.shape[1] + 1, kernel.shape[0]))
    for num in range(kernel.shape[0]):
        print("Filter ", num)
        any_kernel = kernel[num, :]

        if len(any_kernel.shape) > 2:
            conv_map = myconv_(input[:, :, 0], any_kernel[:, :, 0])
            for channel in range(any_kernel.shape[-1]):
                conv_map = conv_map + myconv_(input[:, :, channel], any_kernel[:, :, channel])
        else:
            conv_map = myconv_(input, any_kernel)
        conv_func_output[:, :, num] = conv_map
    return conv_func_output


def maxpooling(ReluFuncRes, dim=2, stride=2):
    maxpool_res = np.zeros((np.uint8((ReluFuncRes.shape[0] - dim + 1) / stride), np.uint8((ReluFuncRes.shape[1] - dim + 1) / stride),
                            ReluFuncRes.shape[-1]))
    for n in range(ReluFuncRes.shape[-1]):
        xx = 0
        for x in np.arange(0, ReluFuncRes.shape[0] - dim - 1, stride):
            yy = 0
            for y in np.arange(0, ReluFuncRes.shape[1] - dim - 1, stride):
                maxpool_res[xx, yy, n] = np.max([ReluFuncRes[x:x + dim, y:y + dim, n]])
                yy = yy + 1
            xx = xx + 1
    return maxpool_res


def Relu_func(conv_res):
    ReluFunc_res = np.zeros(conv_res.shape)
    for n in range(conv_res.shape[-1]):
        for x in np.arange(0, conv_res.shape[0]):
            for y in np.arange(0, conv_res.shape[1]):
                ReluFunc_res[x, y, n] = np.max([conv_res[x, y, n], 0])
    return ReluFunc_res

im = cv2.imread("Data/image/0.png",-1)/255

l1_kernel1 = np.random.rand(4, 3, 3)/10
layer1_conv1 = conv_func(im, l1_kernel1)
layer1_relu1 = Relu_func(layer1_conv1)
l1_kernel2 = np.random.rand(4, 3, 3, layer1_relu1.shape[-1])/10
layer1_conv2 = conv_func(layer1_relu1, l1_kernel2)
layer1_relu2 = Relu_func(layer1_conv2)
layer1_pool = maxpooling(layer1_relu2, 2, 2)

l2_kernel1 = np.random.rand(8, 3, 3, layer1_pool.shape[-1])/10
layer2_conv1 = conv_func(layer1_pool, l2_kernel1)
layer2_relu1 = Relu_func(layer2_conv1)
l2_kernel2 = np.random.rand(8, 3, 3, layer2_relu1.shape[-1])/10
layer2_conv2 = conv_func(layer2_relu1, l2_kernel2)
layer2_relu2 = Relu_func(layer2_conv2)
layer2_pool = maxpooling(layer2_relu2, 2, 2)

l3_kernel1 = np.random.rand(16, 3, 3, layer2_pool.shape[-1])/10
layer3_conv1 = conv_func(layer2_pool, l3_kernel1)
layer3_relu1 = Relu_func(layer3_conv1)
l3_kernel2 = np.random.rand(16, 3, 3, layer3_relu1.shape[-1])/10
layer3_conv2 = conv_func(layer3_relu1, l3_kernel2)
layer3_relu2 = Relu_func(layer3_conv2)
layer3_pool = maxpooling(layer3_relu2, 2, 2)

plt.figure()
plt.imshow(np.c_[np.r_[layer1_conv1[...,0], layer1_conv1[...,1]], np.r_[layer1_conv1[...,2], layer1_conv1[...,3]]], cmap='gray')
plt.savefig('Data\CNNForward\l1conv1_T1')

plt.imshow(np.c_[np.r_[layer1_conv2[...,0], layer1_conv2[...,1]], np.r_[layer1_conv2[...,2], layer1_conv2[...,3]]], cmap='gray')
plt.savefig('Data\CNNForward\l1conv2_T1')

plt.imshow(np.c_[np.r_[layer1_relu2[...,0], layer1_relu2[...,1]], np.r_[layer1_relu2[...,2], layer1_relu2[...,3]]], cmap='gray')
plt.savefig('Data\CNNForward\l1r2_T1')

plt.imshow(np.c_[np.r_[layer1_pool[...,0], layer1_pool[...,1]], np.r_[layer1_pool[...,2], layer1_pool[...,3]]], cmap='gray')
plt.savefig('Data\CNNForward\l1p_T1')

plt.imshow(np.c_[np.r_[layer2_conv2[...,0], layer2_conv2[...,1]], np.r_[layer2_conv2[...,2], layer2_conv2[...,3]]], cmap='gray')
plt.savefig('Data\CNNForward\l2c2_T1')

plt.imshow(np.c_[np.r_[layer2_pool[...,0], layer2_pool[...,1]], np.r_[layer2_pool[...,2], layer2_pool[...,3]]], cmap='gray')
plt.savefig('Data\CNNForward\l2p_T1')

plt.imshow(np.c_[np.r_[layer2_pool[...,0], layer2_pool[...,1]], np.r_[layer2_pool[...,2], layer2_pool[...,3]]], cmap='gray')
plt.savefig('Data\CNNForward\l2p_T1')

plt.imshow(np.c_[np.r_[layer3_pool[...,0], layer3_pool[...,1]], np.r_[layer3_pool[...,2], layer3_pool[...,3]]], cmap='gray')
plt.savefig('Data\CNNForward\l3p_T1')