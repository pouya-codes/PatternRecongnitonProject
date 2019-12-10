# used for caluclating pixel error, dice score, IoU
import numpy as np
def binary(data, thresh):
    data[data > thresh] = 1
    data[data <= thresh] = 0
    return data

def pixel_error(pred, target):
    pix = np.zeros(np.shape(target)[0])
    for k in range(np.shape(target)[0]):
        tptn = np.sum(pred[k].flatten() == target[k].flatten())
        total = np.shape(target)[1] * np.shape(target)[2]
        pix[k] = tptn/total
    pixel_error = pix.mean()
    return pixel_error

def dice_coef (pred, target):
    dice_c1 = np.zeros(np.shape(target)[0])
    dice_c0 = np.zeros(np.shape(target)[0])
    dice = np.zeros(np.shape(target)[0])
    num_pixels = np.shape(target)[1] * np.shape(target)[2]
    for k in range(np.shape(target)[0]):
        # dice one
        both_one = pred[k].flatten() * target[k].flatten()
        intersect_one = both_one.sum()
        dice_one = 2 * intersect_one /(pred[k].sum()+target[k].sum())
        dice_c1[k] = dice_one

        # dice zero
        zero_one = np.sum(np.abs(pred[k].flatten() - target[k].flatten()))
        intersect_zero = num_pixels - intersect_one - zero_one
        dice_zero = 2 * intersect_zero /(2*num_pixels - (pred[k].sum()+target[k].sum()))
        dice_c0[k] = dice_zero

        # dice_average
        dice[k] = (dice_one + dice_zero)/ 2

    dice_c1_mean = dice_c1.mean()
    dice_c0_mean = dice_c0.mean()
    dice_coef = dice.mean()
    return dice_coef, dice_c1_mean, dice_c0_mean

def IoU_coef (pred, target):
    IoU_c1 = np.zeros(np.shape(target)[0])
    IoU_c0 = np.zeros(np.shape(target)[0])
    IoU = np.zeros(np.shape(target)[0])
    num_pixels = np.shape(target)[1] * np.shape(target)[2]
    for k in range(np.shape(target)[0]):
        # IoU class one
        both_one = pred[k].flatten() * target[k].flatten()
        intersect_one = both_one.sum()
        IoU_one = intersect_one /(pred[k].sum()+target[k].sum() - intersect_one)
        IoU_c1[k] = IoU_one

        # IoU class zero
        zero_one = np.sum(np.abs(pred[k].flatten() - target[k].flatten()))
        intersect_zero = num_pixels - intersect_one - zero_one
        IoU_zero = intersect_zero /(2*num_pixels - pred[k].sum() - target[k].sum() - intersect_zero)
        IoU_c0[k] = IoU_zero

        # IoU_average
        IoU[k] = (IoU_one +IoU_zero)/ 2

    IoU_c1_mean = IoU_c1.mean()
    IoU_c0_mean = IoU_c0.mean()
    IoU_coef = IoU.mean()
    return IoU_coef, IoU_c1_mean, IoU_c0_mean



target1 = np.load('test_label.npy')

path = "predict/"

lr_01_s_1 = np.load(f'{path}/lr_0.1_s_1.npy')
lr_01_s_1 = lr_01_s_1[..., -1]

lr_01_s_2 = np.load(f'{path}/lr_0.1_s_2.npy')
lr_01_s_2 = lr_01_s_2[..., -1]

lr_01_s_3 = np.load(f'{path}/lr_0.1_s_3.npy')
lr_01_s_3 = lr_01_s_3[..., -1]

lr_01_s_4 = np.load(f'{path}/lr_0.1_s_4.npy')
lr_01_s_4 = lr_01_s_4[..., -1]

lr_01_s_5 = np.load(f'{path}/lr_0.1_s_5.npy')
lr_01_s_5 = lr_01_s_5[..., -1]




lr_01_s_1 = binary(lr_01_s_1,0.5)
lr_01_s_2 = binary(lr_01_s_2,0.5)
lr_01_s_3 = binary(lr_01_s_3,0.5)
lr_01_s_4 = binary(lr_01_s_4,0.5)
lr_01_s_5 = binary(lr_01_s_5,0.5)



pixelErrorlr_01_s_1 = pixel_error(lr_01_s_1,target1)
pixelErrorlr_01_s_2 = pixel_error(lr_01_s_2,target1)
pixelErrorlr_01_s_3 = pixel_error(lr_01_s_3,target1)
pixelErrorlr_01_s_4 = pixel_error(lr_01_s_4,target1)
pixelErrorlr_01_s_5 = pixel_error(lr_01_s_5,target1)


diceScorelr_01_s_1 = dice_coef(lr_01_s_1, target1)
diceScorelr_01_s_2 = dice_coef(lr_01_s_2, target1)
diceScorelr_01_s_3 = dice_coef(lr_01_s_3, target1)
diceScorelr_01_s_4 = dice_coef(lr_01_s_4, target1)
diceScorelr_01_s_5 = dice_coef(lr_01_s_5, target1)



IoU_Coeflr_01_s_1 = IoU_coef(lr_01_s_1, target1)
IoU_Coeflr_01_s_2 = IoU_coef(lr_01_s_2, target1)
IoU_Coeflr_01_s_3 = IoU_coef(lr_01_s_3, target1)
IoU_Coeflr_01_s_4 = IoU_coef(lr_01_s_4, target1)
IoU_Coeflr_01_s_5 = IoU_coef(lr_01_s_5, target1)

print('pixel_error lr 0.1 s1 : ', pixelErrorlr_01_s_1)
print('pixel_error lr 0.1 s2 : ', pixelErrorlr_01_s_2)
print('pixel_error lr 0.1 s3 : ', pixelErrorlr_01_s_3)
print('pixel_error lr 0.1 s4 : ', pixelErrorlr_01_s_4)
print('pixel_error lr 0.1 s5 : ', pixelErrorlr_01_s_5)

print('Dice coefficient lr 0.1 s1 : ', diceScorelr_01_s_1)
print('Dice coefficient lr 0.1 s2 : ', diceScorelr_01_s_2)
print('Dice coefficient lr 0.1 s3 : ', diceScorelr_01_s_3)
print('Dice coefficient lr 0.1 s4 : ', diceScorelr_01_s_4)
print('Dice coefficient lr 0.1 s5 : ', diceScorelr_01_s_5)

print('IoU coefficient lr 0.1 s1: ', IoU_Coeflr_01_s_1)
print('IoU coefficient lr 0.1 s2: ', IoU_Coeflr_01_s_2)
print('IoU coefficient lr 0.1 s3: ', IoU_Coeflr_01_s_3)
print('IoU coefficient lr 0.1 s4: ', IoU_Coeflr_01_s_4)
print('IoU coefficient lr 0.1 s5: ', IoU_Coeflr_01_s_5)

px01 = [pixelErrorlr_01_s_1, pixelErrorlr_01_s_2, pixelErrorlr_01_s_3, pixelErrorlr_01_s_4, pixelErrorlr_01_s_5]
dice01 = np.array([diceScorelr_01_s_1, diceScorelr_01_s_2, diceScorelr_01_s_3, diceScorelr_01_s_4, diceScorelr_01_s_5])
IoU01 = np.array([IoU_Coeflr_01_s_1, IoU_Coeflr_01_s_2, IoU_Coeflr_01_s_3, IoU_Coeflr_01_s_4, IoU_Coeflr_01_s_5])

print('pixel error mean lr 0.1: ', np.mean(px01))
print('pixel error std lr 0.1: ', np.std(px01))
print('dice score mean lr 0.1: ', dice01.mean(axis=0))
print('dice score std lr 0.1: ',dice01.std(axis=0))
print('IoU mean lr 0.1: ', IoU01.mean(axis=0))
print('IoU std lr 0.1: ', IoU01.std(axis=0))