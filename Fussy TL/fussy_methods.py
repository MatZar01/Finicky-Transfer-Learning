from tfport import *
from imutils import paths
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import InputLayer

"""Set if to recast data to DATA_TYPE
Note that it will make network constructing slower"""
recast_data = True
DATA_TYPE = 'float16'

def list_conv(model):
    """returns list of CONV layers numbers"""
    out = []
    layers = model.layers
    for i in range(0, len(layers)):
        if 'Conv' in str(layers[i]):
            out.append(i)
    return out


def take_n_conv(model, n, conv_layers):
    """returns model with n first conv layers
    BTW, first layer is 0
    input and output feature map shape
    adds Pooling layer before conv if it is in original network"""
    out_model = Sequential()
    try:
        add_pooling = 'MaxPooling2D' in str(model.layers[conv_layers[n] - 1])
    except IndexError:
        add_pooling = False
    # add input layer
    if add_pooling:
        out_model.add(InputLayer(input_shape=model.layers[conv_layers[n] - 1].input_shape[1:]))
        out_model.add(MaxPooling2D(pool_size=(2, 2)))
    else:
        out_model.add(InputLayer(input_shape=model.layers[conv_layers[n]].input_shape[1:]))
    out_model.add(model.layers[conv_layers[n]])
    out_model.compile('SGD', loss='mean_squared_error')
    return out_model, out_model.layers[-1].input_shape[1:], out_model.layers[-1].output_shape[1:]


def load_data(IMAGE_DIR, SEGMENTED_DIR):
    """returns list of tuples:
    (image_path, segmented_path)
    with corresponding names"""
    image_paths = list(paths.list_images(IMAGE_DIR))
    segmented_paths = list(paths.list_images(SEGMENTED_DIR))
    # output list:
    out = []
    # corresponding paths:
    for image_path in image_paths:
        for segmented_path in segmented_paths:
            if image_path.split('/')[-1] == segmented_path.split('/')[-1]:
                out.append((image_path, segmented_path))
                break
    return out


def joint_prediction(model, data_paths_tup, previous_prediction=None, first_prediction=True):
    """returns list of 2 tuples:
    [0]: image prediction
    [1]: manually segmented image resized to prediction"""
    if first_prediction:
        image = cv2.imread(data_paths_tup[0])
        image = np.expand_dims(image, axis=0)
        segmented = cv2.imread(data_paths_tup[1])
    else:
        image = np.empty((1, previous_prediction[0][0].shape[0], previous_prediction[0][0].shape[1], len(previous_prediction[0])))
        for i in range(0, len(previous_prediction[0])):
            image[0, :, :, i] = previous_prediction[0][i]
        #print(image.shape)
        segmented = previous_prediction[1]
    prediction = model.predict(image)
    # output prediction as list of activation maps
    prediction_out = []
    for i in range(0, prediction.shape[-1]):
        prediction_out.append(prediction[0, :, :, i])
    # check if segmented is 0-1 or 0-255 and leave as 0-255
    if segmented.max() < 127:
        segmented = segmented * 255
    # check if segmented has 3 channels, and if so pick only 1
    if segmented.shape[-1] == 3:
        segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    # resize segmented to feature map shape
    segmented = cv2.resize(segmented, prediction.shape[1:3])
    return [prediction_out, segmented]


def IoU_compute(image, segment):
    """compute IoU with prediction and segment"""
    # make sure prediction is in uint8
    image = image.astype('uint8')
    intersection = cv2.bitwise_and(image, segment)
    union = cv2.bitwise_or(image, segment)
    if np.count_nonzero(union) <= 0:
        return 0
    out = np.count_nonzero(intersection) / np.count_nonzero(union)
    return out


def compute_out_matrix(data_list, model, out_shape, previous_prediction=None, first_prediction=True):
    """return complete output matrix for
    selecting best candidate kernels
    with output dimensions:
    images x filters"""
    out = np.empty((len(data_list), out_shape[-1]), dtype='float32')
    # compute prediction for every image
    for i in range(0, len(data_list)):
        if first_prediction:
            prediction = joint_prediction(model, data_list[i])
            #print(len(prediction[0]))
        else:
            prediction = previous_prediction[i]
            print(len(prediction))
        #compute IoU for every prediction
        for j in range(0, len(prediction[0])):
            iou = IoU_compute(prediction[0][j], prediction[1])
            out[i, j] = iou
    return out

def compute_out_matrix_ensemble(data_paths, model, predictions, out_shape):
    out = np.empty((len(predictions), out_shape[-1]), dtype='float32')
    for i in range(0, len(predictions)):
        prediction = predictions[i]
        prediction = joint_prediction(model, data_paths, prediction, False)
        for j in range(0, len(prediction[0])):
            iou = IoU_compute(prediction[0][j], prediction[1])
            out[i, j] = iou
    isnan = True in np.isnan(out)
    if isnan:
        print('[ERROR] NAN in out - model generation might have failed!')
    return out


def compute_filter_mIoU(out_matrix):
    """Returns mean IoU for all of the filters in matrix[a, b]
    where a - image index, b - filter index"""
    output = []
    for i in range(0, out_matrix.shape[1]):
        filter_ious = out_matrix[:, i]
        output.append(np.average(filter_ious))
    return output


def above_average_filters(filters_miou, leave_factor):
    """Returns indices of filters with above average IoU"""
    min_value = np.average(filters_miou)
    list_copy = filters_miou.copy()
    # sort list
    list_copy.sort()
    # reverse list
    list_copy.reverse()
    # get index of last viable item
    index = round((len(list_copy) - 1) * leave_factor)
    min_value = list_copy[index]
    out = []
    for i in range(0, len(filters_miou)):
        if filters_miou[i] >= min_value:
            out.append(i)
    return out


def check_pool_layer(model):
    """returns True if MaxPooling is the next layer"""
    prev_layer = model.layers[0]
    if 'MaxPooling2D' in str(prev_layer):
        return True
    else:
        return False


def construct_network(cut_model, filter_list, input_shape, add_pooling=False, last_pool=False):
    """Returns new network with selected conv kernels"""
    # set data type for network if selected
    if recast_data:
        tf.keras.backend.set_floatx(DATA_TYPE)
    # get conv filters weights, bias and shape
    last_layer = cut_model.layers[-1]
    init_weights = last_layer.get_weights()[0]
    init_bias = last_layer.get_weights()[1]
    shape = list(init_weights.shape[0:-1])
    shape.append(len(filter_list))
    # set build_shape
    build_shape = (1, input_shape[0], input_shape[1], input_shape[2])
    # arrays for filters and bias
    filters = np.empty(shape, dtype='float32')
    bias = np.empty((len(filter_list)), dtype='float32')
    # add weights to new layer
    for i in range(0, len(filter_list)):
        filters[:, :, :, i] = init_weights[:, :, :, filter_list[i]]
        bias[i] = init_bias[filter_list[i]]
    out_layer = [filters, bias]
    # create new model
    out_model = Sequential()
    # add pooling if needed
    if add_pooling:
        out_model.add(MaxPooling2D(pool_size=(2, 2)))
        build_shape = (1, build_shape[1] * 2, build_shape[2] * 2, build_shape[3])
    out_model.add(Conv2D(len(filter_list), (shape[0], shape[1]), padding=last_layer.padding, input_shape=input_shape,
                         name=str(len(init_bias)), activation='relu'))
    if last_pool:
        out_model.add(MaxPooling2D(pool_size=(2, 2)))
    out_model.build(input_shape=build_shape)
    if last_pool:
        out_model.layers[-2].set_weights(out_layer)
    else:
        out_model.layers[-1].set_weights(out_layer)
    out_model.name = lts(filter_list)
    out_model.compile('SGD', loss='mean_squared_error')
    return out_model


def save_model(model, n, conv_layers, out_folder):
    """Saves newly generated model with conv layer number in name
    and _P indicator if pooling is the last layer"""
    out_path = '{}/C_{}.h5'.format(out_folder, conv_layers[n])
    model.save(out_path)
    return True


def predict_filled(model, input_data, seed=314):
    """Run prediction on input data and shuffle it to maintain
    both initial feature space depth and conserved layers position"""
    # set seed for comparable results
    np.random.seed(seed)
    desired_filters = stl(model.name)
    # get conv layer
    for i in range(0, len(model.layers)):
        if "Conv" in str(model.layers[i]):
            break
    desired_output_depth = int(model.layers[i].name)
    kernel_map = np.random.randint(len(desired_filters), size=desired_output_depth)
    # make prediction
    prediction = model.predict(input_data)
    # make output feature space
    output = np.empty((prediction.shape[0], prediction.shape[1], prediction.shape[2], desired_output_depth),
                      dtype='float32')
    # insert predictions to output in correct order
    for i in range(0, desired_output_depth):
        if i in desired_filters:
            output[0, :, :, i] = prediction[0, :, :, desired_filters.index(i)]
        else:
            output[0, :, :, i] = prediction[0, :, :, kernel_map[i]]
    return output


def add_pooling(model):
    """Adds 2x2 pooling layers at the end of the model"""
    out_model = Sequential()
    for layer in model.layers:
        out_model.add(layer)
    out_model.add(MaxPooling2D(pool_size=(2, 2)))
    out_model.compile('SGD', loss='mean_squared_error')
    return out_model


def predict_from_model_parts(model_list, data_paths_list, seed=314):
    """Returns last joint prediction from last model from workflow list
    combined with segmented image with correct size"""
    output = []
    for data_tuple in data_paths_list:
        image = cv2.imread(data_tuple[0])
        prediction = np.expand_dims(image, axis=0)
        segmented = cv2.imread(data_tuple[1])
        # normalize segmented
        if segmented.max() < 127:
            segmented = segmented * 255
        # check if segmented has 3 channels, and if so pick only 1
        if segmented.shape[-1] == 3:
            segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
        prediction_out = []
        for m in range(0, len(model_list)):
            prediction = predict_filled(model_list[m], prediction, seed)
            if m == len(model_list) - 1:
                for i in range(0, prediction.shape[-1]):
                    prediction_out.append(prediction[0, :, :, i])
        output.append([prediction_out, cv2.resize(segmented, prediction.shape[1:3])])
    return output


def extract_features_from_parts(model_list, data, seed=314):
    """Returns last joint prediction from last model from workflow list
    combined with segmented image with correct size"""
    output = []
    # assess data
    if len(data.shape) == 3:
        prediction = np.expand_dims(data, axis=0)
    else:
        prediction = data
    # check if segmented has 3 channels, and if so pick only 1
    for m in range(0, len(model_list)):
        prediction = predict_filled(model_list[m], prediction, seed)
        if m == len(model_list) - 1:
            return prediction


def simple_verbose(i, conv_list=None, filter_list=None):
    """Returns status verbose"""
    if conv_list == None and filter_list == None:
        print('[INFO] Model Generation in progress')
    if conv_list != None:
        print('[INFO] CONV layer {} with input {} and output {} complete and saved!'.format(conv_list[i][0],
                                                                                            conv_list[i][1],
                                                                                            conv_list[i][2]))
    if filter_list != None:
        print('[INFO] {} filters from CONV {} selected'.format(len(filter_list), i))
