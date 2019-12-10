'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is the main file for the project. From here you can train, test,
    and manipulate the SegCaps of models.
Please see the README for detailed instructions for this project.

==============
This is the entry point of the package to train UNet, tiramisu,
    Capsule Nets (capsbasic) or SegCaps(segcapsr1 or segcapsr3).

@author: Cheng-Lin Li a.k.a. Clark

@copyright:2018 Cheng-Lin Li@Insight AI. All rights reserved.

@license:  Licensed under the Apache License v2.0.
            http://www.apache.org/licenses/

@contact:    clark.cl.li@gmail.com

Tasks:
    The program load parameters for training, testing, manipulation
        for all models.


Data:
    MS COCO 2017 or LUNA 2016 were tested on this package.
    You can leverage your own data set but the mask images should follow the format of MS COCO or with background color = 0 on each channel.

Enhancement: 
  1. The program was modified to support python 3.6 on Ubuntu 18.04 and Windows 10.
  2. Support not only 3D computed tomography scan images but also 2D Microsoft Common Objects in COntext (MS COCO) dataset images.
  3. Add Kfold parameter for users to customize the cross validation task. K = 1 will force model to perform overfit.
  4. Add retrain parameter to enable users to reload pre-trained weights and retrain the model.
  5. Add initial learning rate for users to adjust.
  6. Add steps per epoch for users to adjust.
  7. Add number of patience for early stop of training to users.
  8. Add 'bce_dice' loss function as binary cross entropy + soft dice coefficient.
  9. Revise 'train', 'test', 'manip' flags from 0 or 1 to flags show up or not to indicate the behavior of main program.
'''

from __future__ import print_function
import sys
import logging
import platform
from os.path import join
from os import makedirs
from os import environ
import argparse
import SimpleITK as sitk  # image process
from time import gmtime, strftime
from keras.utils import print_summary
from utils.load_data import load_data, split_data
from utils.model_helper import create_model

time = strftime("%Y%m%d-%H%M%S", gmtime())
RESOLUTION = 512  # Resolution of the input for the model.
GRAYSCALE = True
LOGGING_FORMAT = '%(levelname)s %(asctime)s: %(message)s'


def main(args):
    # Ensure training, testing, and manip are not all turned off
    assert (args.train or args.test or args.manip), 'Cannot have train, test, and manip all set to 0, Nothing to do.'

    # Load the training, validation, and testing data
    try:
        train_list, val_list, test_list = load_data(args.data_root_dir, args.split_num)
    except:
        # Create the training and test splits if not found
        logging.info('\nNo existing training, validate, test files...System will generate it.')
        split_data(args.data_root_dir, num_splits=args.Kfold)
        train_list, val_list, test_list = load_data(args.data_root_dir, args.split_num)

    # Get image properties from first image. Assume they are all the same.
    logging.info('\nRead image files...%s' % (join(args.data_root_dir, 'imgs', train_list[0][0])))
    # Get image shape from the first image.
    image = sitk.GetArrayFromImage(sitk.ReadImage(join(args.data_root_dir, 'imgs', train_list[0][0])))
    img_shape = image.shape  # # (x, y, channels)

    # if args.dataset == 'luna16':
    #     net_input_shape = (img_shape[1], img_shape[2], args.slices)
    # else:
    #     args.slices = 1
    if GRAYSCALE == True:
        net_input_shape = (img_shape[0], img_shape[1], 1)  # only one channel

    else:
        net_input_shape = (img_shape[0], img_shape[1], 3)  # Only access RGB 3 channels.

    # Create the model for training/testing/manipulation
    # enable_decoder = False only for SegCaps R3 to disable recognition image output on evaluation model 
    # to speed up performance.
    model_list = create_model(args=args, input_shape=net_input_shape, enable_decoder=True)
    print_summary(model=model_list[0], positions=[.38, .65, .75, 1.])

    args.output_name = 'split-' + str(args.split_num) + '_batch-' + str(args.batch_size) + \
                       '_shuff-' + str(args.shuffle_data) + '_aug-' + str(args.aug_data) + \
                       '_loss-' + str(args.loss) + '_slic-' + str(args.slices) + \
                       '_sub-' + str(args.subsamp) + '_strid-' + str(args.stride) + \
                       '_lr-' + str(args.initial_lr) + '_recon-' + str(args.recon_wei)

    #     args.output_name = 'sh-' + str(args.shuffle_data) + '_a-' + str(args.aug_data)

    args.time = time
    if platform.system() == 'Windows':
        args.use_multiprocessing = False
    else:
        args.use_multiprocessing = True
    args.check_dir = join(args.data_root_dir, 'saved_models', args.net)
    try:
        makedirs(args.check_dir)
    except:
        pass

    args.log_dir = join(args.data_root_dir, 'logs', args.net)
    try:
        makedirs(args.log_dir)
    except:
        pass

    args.tf_log_dir = join(args.log_dir, 'tf_logs')
    try:
        makedirs(args.tf_log_dir)
    except:
        pass

    args.output_dir = join(args.data_root_dir, 'plots', args.net)
    try:
        makedirs(args.output_dir)
    except:
        pass

    if args.train == True:
        from train import train
        # Run training
        train(args, train_list, val_list, model_list[0], net_input_shape)

    if args.test == True:
        from test import test
        # Run testing
        test(args, test_list, model_list, net_input_shape)

    if args.manip == True:
        from manip import manip
        # Run manipulation of segcaps
        manip(args, test_list, model_list, net_input_shape)


def loglevel(level=0):
    assert isinstance(level, int)
    try:
        return [
            # logging.CRITICAL,
            # logging.ERROR,
            logging.WARNING,  # default
            logging.INFO,
            logging.DEBUG,
            logging.NOTSET,
        ][level]
    except LookupError:
        return logging.NOTSET


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train on Medical Data or MS COCO dataset'
    )
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory for your data.')
    parser.add_argument('--weights_path', type=str, default='',
                        help='/path/to/trained_model.hdf5 from root. Set to "" for none.')
    parser.add_argument('--split_num', type=int, default=0,
                        help='Which training split to train/test on.')
    parser.add_argument('--net', type=str.lower, default='segcapsr3',
                        choices=['segcapsr3', 'segcapsr1', 'capsbasic', 'unet', 'tiramisu'],
                        help='Choose your network.')
    parser.add_argument('--train', action='store_true',
                        help='Add this flag to enable training.')
    parser.add_argument('--test', action='store_true',
                        help='Add this flag to enable testing.')
    parser.add_argument('--manip', action='store_true',
                        help='Add this flag to enable manipulation.')
    parser.add_argument('--shuffle_data', type=int, default=1, choices=[0, 1],
                        help='Whether or not to shuffle the training data (both per epoch and in slice order.')
    parser.add_argument('--aug_data', type=int, default=1, choices=[0, 1],
                        help='Whether or not to use data augmentation during training.')
    parser.add_argument('--loss', type=str.lower, default='w_bce',
                        choices=['bce', 'w_bce', 'dice', 'bce_dice', 'mar', 'w_mar'],
                        help='Which loss to use. "bce" and "w_bce": unweighted and weighted binary cross entropy'
                             ', "dice": soft dice coefficient, "bce_dice": binary cross entropy + soft dice coefficient'
                             ', "mar" and "w_mar": unweighted and weighted margin loss.')
    # TODO: multiclass segmentation.
    #   # Calculate distance from actual labels using cross entropy
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    #   #Take mean for total loss
    # loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")    
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training/testing.')
    parser.add_argument('--initial_lr', type=float, default=0.00001,
                        help='Initial learning rate for Adam.')
    parser.add_argument('--steps_per_epoch', type=int, default=1000,
                        help='Number of iterations in an epoch.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for training.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of patience indicates the criteria of early stop training.'
                             'If score of metrics do not improve during the patience of epochs,'
                             ' the training will be stopped.')
    parser.add_argument('--recon_wei', type=float, default=131.072,
                        help='If using capsnet: The coefficient (weighting) for the loss of decoder')
    parser.add_argument('--slices', type=int, default=1,
                        help='Number of slices to include for training/testing.')
    parser.add_argument('--subsamp', type=int, default=-1,
                        help='Number of slices to skip when forming 3D samples for training. Enter -1 for random '
                             'subsampling up to 5%% of total slices.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Number of slices to move when generating the next sample.')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Set the verbose value for training. 0: Silent, 1: per iteration, 2: per epoch.')
    parser.add_argument('--save_raw', type=int, default=1, choices=[0, 1],
                        help='Enter 0 to not save, 1 to save.')
    parser.add_argument('--save_seg', type=int, default=1, choices=[0, 1],
                        help='Enter 0 to not save, 1 to save.')
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Prefix to append to saved CSV.')
    parser.add_argument('--thresh_level', type=float, default=0.,
                        help='Enter 0.0 for masking refine by Otsu algorithm.'
                             ' Or set a value for thresholding level of masking. Value should between 0 and 1.')
    parser.add_argument('--compute_dice', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_jaccard', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_assd', type=int, default=0,
                        help='0 or 1')
    parser.add_argument('--which_gpus', type=str, default='0',
                        help='Enter "-2" for CPU only, "-1" for all GPUs available, '
                             'or a comma separated list of GPU id numbers ex: "0,1,4".')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs you have available for training. '
                             'If entering specific GPU ids under the --which_gpus arg or if using CPU, '
                             'then this number will be inferred, else this argument must be included.')
    # Enhancements: 
    # TODO: implement softmax entroyp loss function for multiclass segmentation
    parser.add_argument('--dataset', type=str.lower, default='mscoco17', choices=['luna16', 'mscoco17'],
                        help='Enter "mscoco17" for COCO dataset, "luna16" for CT images')
    parser.add_argument('--num_class', type=int, default=2,
                        help='Number of classes to segment. Default is 2. If number of classes > 2, '
                             ' the loss function will be softmax entropy and only apply on SegCapsR3'
                             '** Current version only support binary classification tasks.')
    parser.add_argument('--Kfold', type=int, default=4, help='Define K value for K-fold cross validate'
                                                             ' default K = 4, K = 1 for over-fitting test')
    parser.add_argument('--retrain', type=int, default=0, choices=[0, 1],
                        help='Retrain your model based on existing weights.'
                             ' default 0=train your model from scratch, 1 = retrain existing model.'
                             ' The weights file location of the model has to be provided by --weights_path parameter')
    parser.add_argument('--loglevel', type=int, default=4, help='loglevel 3 = debug, 2 = info, 1 = warning, '
                                                                ' 4 = error, > 4 =critical')
    arguments = parser.parse_args()

    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    logging.basicConfig(format=LOGGING_FORMAT, level=loglevel(arguments.loglevel), stream=sys.stderr)

    if arguments.which_gpus == -2:
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = ""
    elif arguments.which_gpus == '-1':
        assert (
                    arguments.gpus != -1), 'Use all GPUs option selected under --which_gpus, with this option the user MUST ' \
                                           'specify the number of GPUs available with the --gpus option.'
    else:
        arguments.gpus = len(arguments.which_gpus.split(','))
        environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = str(arguments.which_gpus)

    if arguments.gpus > 1:
        assert arguments.batch_size >= arguments.gpus, 'Error: Must have at least as many items per batch as GPUs ' \
                                                       'for multi-GPU training. For model parallelism instead of ' \
                                                       'data parallelism, modifications must be made to the code.'

    main(arguments)
