# from numpy import load
# import numpy as np
# import cv2
# import sys
# # np.set_printoptions(threshold=sys.maxsize)
# from utils.custom_data_aug import custom_background_by_us
# #
# data = load('train3.npz')
#
# lst = data.files
# for item in lst:
#     print(item)
#     print(data[item])
#     cv2.imshow('Im', data[item])
#     cv2.waitKey(0)
#
#
#
# # a = np.arange(25,dtype='int').reshape(5,5,1)
# # print(a)
# # b = custom_background_by_us(a)
# # print(b)
#
#
#
#
# img=cv2.imread('train2.png')
# print("Shape of mask ",img.shape)
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # gray = gray.reshape(gray.shape[0],gray.shape[1],1)
# # print(gray.shape)
# # cv2.imshow('gray', gray)
# # img = np.array(img)
# # print(img.shape)
# # cv2.imshow('original',img)
# # cv2.waitKey(0)
#
# # out = cv2.subtract(255, img)
# # print('Output Shae',out.shape)
# # print('Gray Shape',gray.shape)
# # cv2.imshow('Outour', out)
# # cv2.waitKey(0)
# # im = custom_background_by_us(gray)
# # cv2.imshow('Converted', im)
# # cv2.waitKey(0)