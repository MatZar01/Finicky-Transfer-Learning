from tfport import *
import timport
from keras.applications import VGG16
import fussy_methods

# Set dirs
IMAGE_DIR = './Images'
SEGMENTED_DIR = './Segmented'
OUT_DIR = './model_parts'

# Set input size
input_size = (224, 224, 3)

# select pseudo-random seed
SEED = 314

# set classes number
CLASSES = 2

# set average threshold
leave_factor = 0.17

# load backbone model
backbone_model = VGG16(weights='imagenet', include_top=False, input_shape=input_size)
backbone_parameters = backbone_model.count_params()

# export conv layers numbers
conv_layers = fussy_methods.list_conv(backbone_model)

# load list of image-segmented path tuples
data_paths_list = fussy_methods.load_data(IMAGE_DIR, SEGMENTED_DIR)

# # take model ending with n-th conv
# cut_model, in_shape, out_shape = fussy_methods.take_n_conv(backbone_model, 0, conv_layers)
#
# # check prev layer
# add_pooling = fussy_methods.check_next_layer(backbone_model, 0, conv_layers)
#
# # provide output matrix
# output_matrix = fussy_methods.compute_out_matrix(data_paths_list, cut_model, out_shape)
#
# # compute filters IoUs
# filters_miou = fussy_methods.compute_filter_mIoU(output_matrix)
#
# # find only filters above average
# above_avg_filters = fussy_methods.above_average_filters(filters_miou)
#
# # with known above avg filters, construct new conv model with mapped outputs
# new_model = fussy_methods.construct_network(cut_model, above_avg_filters, in_shape, add_pooling)

# complete workflow
# take backbone model and slice it into conv layesr
conv_list = []
pruned_models = []
for n in range(0, len(conv_layers)):
    conv_list.append(fussy_methods.take_n_conv(backbone_model, n, conv_layers))

# prune each model layer
for i in range(0, len(conv_list)):
    # case -1: add last pool layer
    if i == len(conv_list) - 1:
        last_pool = True
    else:
        last_pool = False

    # case 0: first model in list
    if i == 0:
        # select best kernels
        output_matrix = fussy_methods.compute_out_matrix(data_paths_list, conv_list[i][0], conv_list[i][-1])
        filters_miou = fussy_methods.compute_filter_mIoU(output_matrix)
        above_avg_filters = fussy_methods.above_average_filters(filters_miou, leave_factor)

        # verbose
        fussy_methods.simple_verbose(i, filter_list=above_avg_filters)

        # build new model
        new_model = fussy_methods.construct_network(conv_list[i][0], above_avg_filters, conv_list[i][1], False)
        pruned_models.append(new_model)


    # case 1: regular operation with pruned networks
    else:
        # get predictions from previous layers
        predictions = fussy_methods.predict_from_model_parts(pruned_models, data_paths_list, 314)
        # select best kernels
        output_matrix = fussy_methods.compute_out_matrix_ensemble(data_paths_list, conv_list[i][0], predictions, conv_list[i][-1])
        filters_miou = fussy_methods.compute_filter_mIoU(output_matrix)
        above_avg_filters = fussy_methods.above_average_filters(filters_miou, leave_factor)

        # verbose
        fussy_methods.simple_verbose(i, filter_list=above_avg_filters)

        #build new model
        add_pooling = fussy_methods.check_pool_layer(conv_list[i][0])
        new_model = fussy_methods.construct_network(conv_list[i][0], above_avg_filters, conv_list[i][1], add_pooling, last_pool)
        pruned_models.append(new_model)

    # save model
    fussy_methods.save_model(new_model, i, conv_layers, OUT_DIR)

    # verbose
    fussy_methods.simple_verbose(i, conv_list=conv_list)

print('[INFO] Counting parameters')
params = 0
for model in pruned_models:
    params += model.count_params()

print('[INFO] Total model parameters: {}'.format(params))
print('[INFO] {}% parameters pruned'.format(round(1 - (params/backbone_parameters), 3)))

print("All done!")
