from tfport import *
import timport
import numpy as np
from keras.applications import VGG16
import fussy_methods
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import load_model
from keras.layers import BatchNormalization
import sys

# load images dirs
IMAGES_DIR_CRACK = './Images/Crack'
SEGMENTED_DIR_CRACK = './Segmented/Crack'
IMAGES_DIR_BACKGROUND = './Images/Background'
SEGMENTED_DIR_BACKGROUND = './Segmented/Background'

out_def = 'models'
bone_def = './backbones/alex3_base.h5'

# get run arguments
args = sys.argv

if len(args) == 1:
    out_dir = out_def
    backbone_path = bone_def
    total_models = 1.00
else:
    out_dir = args[1]
    backbone_path = args[2]
    total_models = float(args[3]) / 100

# load backbone model
input_shape = (1, 224, 224, 3)
backbone_input_shape = (224, 224, 3)
backbone = load_model(backbone_path)
# list conv layers
conv_layers_num = fussy_methods.list_conv(backbone)

# load image-segmented tuples
data_path_list_crack = fussy_methods.load_data(IMAGES_DIR_CRACK, SEGMENTED_DIR_CRACK)
data_path_list_background = fussy_methods.load_data(IMAGES_DIR_BACKGROUND, SEGMENTED_DIR_BACKGROUND)

# iterator as leave_factor
ITERATOR = 0.01
# iterate every layer in backbone
while True:
    # add output model and initial temp model
    out_model = Sequential()
    temp_model = Sequential()
    weights = []
    
    # iterate every layer in backbone
    for layer in backbone.layers:
        if 'Input' in str(layer):
            continue
        # case 0: non-conv layer
        if 'MaxPooling' in str(layer):
            # get pool size
            temp_model.add(MaxPooling2D(pool_size=layer.pool_size, strides=layer.strides))
            out_model.add(MaxPooling2D(pool_size=layer.pool_size, strides=layer.strides))
            temp_model.build(input_shape=input_shape)
            continue
        if 'BatchNormalization' in str(layer):
            temp_model.add(BatchNormalization())
            out_model.add(BatchNormalization())
            temp_model.build(input_shape=input_shape)
            continue
        # case 1: first conv layer
        if backbone.layers.index(layer) == conv_layers_num[0]:
            filters = len(layer.get_weights()[1])
            temp_model.add(layer)
            temp_model.build(input_shape=input_shape)
            # make prediction for every data tuple and get out matrix
            output_matrix_crack = fussy_methods.compute_out_matrix_c(data_path_list_crack, temp_model)
            output_matrix_background = fussy_methods.compute_out_matrix_c(data_path_list_background, temp_model)
            # assess prediction
            filters_miou_crack = fussy_methods.compute_filter_mIoU(output_matrix_crack)
            filters_miou_background = fussy_methods.compute_filter_mIoU(output_matrix_background)
            # get above avg filters
            above_avg_filters_crack = fussy_methods.above_average_filters(filters_miou_crack, ITERATOR, filters)
            #above_avg_filters_background = fussy_methods.above_average_filters(filters_miou_background, ITERATOR, filters)
            above_avg_filters_background = []
            above_avg_filters = fussy_methods.merge_lists(above_avg_filters_background, above_avg_filters_crack)
            # get filters to remove
            filters_to_remove = fussy_methods.filters_to_remove(filters, above_avg_filters)
            # build new conv with purged filters and add to out model
            conv = temp_model.layers[-1]
            filters = conv.get_weights()[0]
            bias = conv.get_weights()[1]
            for filter in filters_to_remove:
                filters = np.delete(filters, filter, 3)
                bias = np.delete(bias, filter, 0)
            new_conv = Conv2D(len(bias), layer.kernel_size, padding='same', activation='relu', strides=layer.strides)
            out_model.add(new_conv)
            weights.append([filters, bias])
            out_model.build(input_shape=input_shape)
            out_model.layers[-1].set_weights(weights[-1])
            #print(out_model.summary())
        # case regular: conv layer in middle of network
        else:
            filters = len(layer.get_weights()[1])
            last = False
            if backbone.layers.index(layer) == conv_layers_num[-1]:
                last = True
            # first purge kernels from every filter
            purged_layer = fussy_methods.purge_kernels(layer, filters_to_remove)
            # construct new purged layer
            temp_model.add(purged_layer[0])
            temp_model.build(input_shape=(1, 224, 224, 3))
            temp_model.layers[-1].set_weights(purged_layer[1])
            # now do regular filter purging
            # make prediction for every data tuple and get out matrix
            output_matrix_crack = fussy_methods.compute_out_matrix_c(data_path_list_crack, temp_model)
            output_matrix_background = fussy_methods.compute_out_matrix_c(data_path_list_background,
                                                                                       temp_model)
            # assess prediction
            filters_miou_crack = fussy_methods.compute_filter_mIoU(output_matrix_crack)
            filters_miou_background = fussy_methods.compute_filter_mIoU(output_matrix_background)
            # get above avg filters
            above_avg_filters_crack = fussy_methods.above_average_filters(filters_miou_crack, ITERATOR, filters)
            #above_avg_filters_background = fussy_methods.above_average_filters(filters_miou_background, ITERATOR, filters)
            above_avg_filters_background = []
            above_avg_filters = fussy_methods.merge_lists(above_avg_filters_background, above_avg_filters_crack)
            # get filters to remove
            filters_to_remove = fussy_methods.filters_to_remove(filters, above_avg_filters)
            # build new conv with purged filters and add to out model
            conv = temp_model.layers[-1]
            filters = conv.get_weights()[0]
            bias = conv.get_weights()[1]
            for filter in filters_to_remove:
                filters = np.delete(filters, filter, 3)
                bias = np.delete(bias, filter, 0)
            new_conv = Conv2D(len(bias), layer.kernel_size, padding='same', activation='relu', strides=layer.strides)
            out_model.add(new_conv)
            print(temp_model.summary())
            weights.append([filters, bias])
            out_model.build(input_shape=input_shape)
            out_model.layers[-1].set_weights(weights[-1])
            weights.append([filters, bias])
        # after purging -> temp model becomes briefly out model
        temp_model = fussy_methods.copy_model(out_model)
    
    # build model
    out_model.build(input_shape=input_shape)

    print(out_model.summary())
    
    # test model
    im = np.zeros((1, 224, 224, 3))
    ft = out_model.predict(im)
    out_model.compile('SGD', loss='mean_squared_error')
    out_model.save('./{}/model_{}_al.h5'.format(out_dir, round(ITERATOR, 3)))
    print(ft.shape)
    ITERATOR += 0.01
    if ITERATOR >= total_models:
        break

print("DONE")
