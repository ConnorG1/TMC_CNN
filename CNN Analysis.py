# Dependencies
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from scipy import interp

import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
import matplotlib.pyplot as plt

from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

from sklearn.metrics import plot_confusion_matrix

def extract_filenames(file_list, gene, channel, seed, limit):
    """Extract filenames containing a given gene from a file list"""

    filtered_fnames = []
    print(f'Finding {gene} {channel} files...')
    for file in file_list:
        if gene in file:
            filtered_fnames.append(file)
    if len(filtered_fnames) >= limit:
        random.seed(seed)
        filtered_fnames = random.sample(filtered_fnames, limit)

    return filtered_fnames


def replace_channel(file_list, new_channel):
    """Replaces channel name in file string - assumes that the original channel is AGP"""

    replaced_filenames = []
    for file in file_list:
        file = file.replace('AGP', new_channel)
        replaced_filenames.append(file)

    return replaced_filenames


def get_filenames(csv, gene_list, channel, seed, limit):
    """Extracts filenames from CSV file, replaces channel string and returns a dictionary"""

    # Read csv
    df = pd.read_csv(csv)
    filenames = df['File'].to_list()  # Filename header must be called 'File'

    gene_filenames = {}
    for gene in gene_list:
        gene_files = extract_filenames(file_list=filenames, gene=gene, channel = channel, seed=seed, limit = limit)
        if channel == 'AGP':
            gene_files = replace_channel(file_list=gene_files, new_channel='AGP')
        if channel == 'ER':
            gene_files = replace_channel(file_list=gene_files, new_channel='ER')
        if channel == 'Mito':
            gene_files = replace_channel(file_list=gene_files, new_channel='Mito')
        if channel == 'Nucleus':
            gene_files = replace_channel(file_list=gene_files, new_channel='Nucleus')
        if channel == 'Syto':
            gene_files = replace_channel(file_list=gene_files, new_channel='Syto')
        gene_filenames[gene] = gene_files

    return gene_filenames


def process_load_image(filename, rotate, flip):
    """Loads image from filename, resizes, rotates (if necessary) converts to RGB"""

    img = Image.open(filename).resize((224, 224))
    if rotate == 90:
        img = img.rotate(90)
    if rotate == 180:
        img = img.rotate(180)
    if rotate == 270:
        img = img.rotate(270)
    if flip == 'LR':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = np.asarray(img)
    img = np.stack((img,)*3, axis=-1)
    img = img / 255.0
    
    return img


def load_and_rotate(filename_dict, channel, limit, image_folder='C:\\Users\\Connor\\Desktop\\Research\\TMC_final\\'):
    """Takes a dictionary, loads the images and balances the smaller set by rotating"""
    
    gene_array_dict = {}
    gene_labels_dict = {}

    for gene, label in zip(filename_dict, range(len(filename_dict))):
        
        if len(filename_dict[gene]) == limit: # Load size 3000 images
            gene_array = []
            gene_labels = []
            for file in filename_dict[gene]:
                #print('Loading: ', file)
                img = process_load_image(os.path.join(image_folder, channel, file), rotate=None, flip=None)
                gene_array.append(img)
                gene_labels.append(float(label))
                if len(gene_array) % 100 == 0:
                    print(f'Processed {len(gene_array)} / {len(filename_dict[gene])} {gene} images')
            
            gene_array_dict[gene] = gene_array
            gene_labels_dict[gene] = gene_labels

        if len(filename_dict[gene]) < limit: # Load all less than 3000
            gene_array = []
            gene_labels = []
            for file in filename_dict[gene]:
                img = process_load_image(os.path.join(image_folder, channel, file), rotate=None, flip=None)
                gene_array.append(img)
                gene_labels.append(float(label))
                if len(gene_array) % 100 == 0:
                    print(f'Processed {len(gene_array)} / {len(filename_dict[gene])} {gene} images')

            # Rotate 90 degrees
            for file in filename_dict[gene]:
                if len(gene_array) < limit:
                    img = process_load_image(os.path.join(image_folder, channel, file), rotate=90, flip=None)
                    gene_array.append(img)
                    gene_labels.append(float(label))
                else:
                    break
                if len(gene_array) % 100 == 0:
                    print(f'Processed {len(gene_array)} / {len(filename_dict[gene])} {gene} images - 90 degree rotation')

            # Rotate 180 degrees
            for file in filename_dict[gene]:
                if len(gene_array) < limit:
                    img = process_load_image(os.path.join(image_folder, channel, file), rotate = 180, flip = None)
                    gene_array.append(img)
                    gene_labels.append(float(label))
                else:
                    break
                if len(gene_array) % 100 == 0:
                    print(f'Processed {len(gene_array)} / {len(filename_dict[gene])} {gene} images - 180 degree rotation')
            
           # Rotate 270 degrees
            for file in filename_dict[gene]:
                if len(gene_array) < limit:
                    img = process_load_image(os.path.join(image_folder, channel, file), rotate = 270, flip = None)
                    gene_array.append(img)
                    gene_labels.append(float(label))
                else:
                    break
                if len(gene_array) % 100 == 0:
                    print(f'Processed {len(gene_array)} / {len(filename_dict[gene])} {gene} images - 270 degree rotation')
                    
            # Horizontal flip with no rotation
            for file in filename_dict[gene]:
                if len(gene_array) < limit:
                    img = process_load_image(os.path.join(image_folder, channel, file), rotate = None, flip = 'LR')
                    gene_array.append(img)
                    gene_labels.append(float(label))
                else:
                    break
                if len(gene_array) % 100 == 0:
                    print(f'Processed {len(gene_array)} / {len(filename_dict[gene])} {gene} images - left-right flip')

            # Horizontal flip with 90 degrees rotation
            for file in filename_dict[gene]:
                if len(gene_array) < limit:
                    img = process_load_image(os.path.join(image_folder, channel, file), rotate = 90, flip = 'LR')
                    gene_array.append(img)
                    gene_labels.append(float(label))
                else:
                    break
                if len(gene_array) % 100 == 0:
                    print(f'Processed {len(gene_array)} / {len(filename_dict[gene])} {gene} images - left-right flip 90 degree rotation')

            # Horizontal flip with 180 degrees rotation
            for file in filename_dict[gene]:
                if len(gene_array) < limit:
                    img = process_load_image(os.path.join(image_folder, channel, file), rotate = 180, flip = 'LR')
                    gene_array.append(img)
                    gene_labels.append(float(label))
                else:
                    break
                if len(gene_array) % 100 == 0:
                    print(f'Processed {len(gene_array)} / {len(filename_dict[gene])} {gene} images - left-right flip 180 degree rotation')

            # Horizontal flip with 270 degrees rotation
            for file in filename_dict[gene]:
                if len(gene_array) < limit:
                    img = process_load_image(os.path.join(image_folder, channel, file), rotate = 270, flip = 'LR')
                    gene_array.append(img)
                    gene_labels.append(float(label))
                else:
                    break
                if len(gene_array) % 100 == 0:
                    print(f'Processed {len(gene_array)} / {len(filename_dict[gene])} {gene} images - left-right flip 270 degree rotation')
            
            gene_array_dict[gene] = gene_array
            gene_labels_dict[gene] = gene_labels
    
    image_array_size = []
    for gene in gene_array_dict:
        image_array_size.append(len(gene_array_dict[gene]))
    smallest_size = min(image_array_size)

    for gene, label in zip(gene_array_dict, gene_labels_dict): # Prune back dataset
        gene_array_dict[gene] = gene_array_dict[gene][:smallest_size]
        gene_labels_dict[label] = gene_labels_dict[label][:smallest_size]

    combined_array = []
    for gene in gene_array_dict: # Merge arrays into one list
        combined_array.append(gene_array_dict[gene])

    final_array = np.concatenate(combined_array)

    final_labels = []   
    for labels in gene_labels_dict: # Append labels
        for x in gene_labels_dict[labels]:
            final_labels.append(x)
    
    final_labels_combined = np.asarray(final_labels) # Turn into an array
    final_labels_combined = final_labels_combined.reshape(-1, 1) # Reshape into 2D array (many rows, one column)
    cat = OneHotEncoder() # Initialise OneHotEncode
    ohe = cat.fit_transform(final_labels_combined).toarray() # Perform one hot encoding and return array

    for x in gene_array_dict:
        gene_array_dict[x] = np.stack(gene_array_dict[x], axis = 0)
        print(f'{x} array shape: ', gene_array_dict[x].shape)
    print(f'\nTotal array shape: ', final_array.shape)
    print(f'Labels array shape: ', final_labels_combined.shape)
    
    return final_array, ohe

def generate_data(gene_list, channel, seed, limit):
    
    csv = "C://Users//Connor//Desktop//TMC_Final//Tags.csv" # Set csv path
    
    filename_dict = get_filenames(csv=csv, gene_list = gene_list, channel=channel, seed=seed, limit = limit)
    array, labels = load_and_rotate(filename_dict = filename_dict, channel = channel, limit = limit)

    X_train, X_remain, y_train, y_remain = train_test_split(array, labels, test_size=0.2) # Split data
    X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=0.5)
    
    print('\nTraining data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('\nValidation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('\nTest data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape, '\n')
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_model(classes):
    
    """Creates standard AlexNet CNN"""
    
    # Instantiate an empty model
    model = Sequential()

    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.summary()

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model

def generate_confusion_matrix(y_test, y_pred, gene_list):

    """Compute confuusion matrix"""

    # Generate data
    class_names = np.array(gene_list)

    file_name = '_'.join(gene_list)

    conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    conf_mat_df = pd.DataFrame(data=conf_mat, index=gene_list, columns=gene_list)
    conf_mat_df.columns = pd.MultiIndex.from_product([['Pred'], conf_mat_df.columns])
    conf_mat_df.index = pd.MultiIndex.from_product([['True'], conf_mat_df.index])

    return conf_mat_df

def compute_auc(y_pred, y_test, gene_list):

    """Compute AUC given a prediction list and test list"""

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(gene_list)
    for gene, i in zip(gene_list, range(n_classes)):
        fpr[f'fpr_{gene}'], tpr[f'tpr_{gene}'], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[f'auc_{gene}'] = auc(fpr[f'fpr_{gene}'], tpr[f'tpr_{gene}'])

    # Compute micro-average ROC curve and ROC area
    fpr["fpr_micro"], tpr["tpr_micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["auc_micro"] = auc(fpr["fpr_micro"], tpr["tpr_micro"])
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[f'fpr_{gene}'] for i in gene_list]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for gene in gene_list:
        mean_tpr += np.interp(all_fpr, fpr[f'fpr_{gene}'], tpr[f'tpr_{gene}'])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["fpr_macro"] = all_fpr
    tpr["tpr_macro"] = mean_tpr
    roc_auc["auc_macro"] = auc(fpr["fpr_macro"], tpr["tpr_macro"])

    # Combine into one dataframe

    combined = {}
    combined.update(fpr)
    combined.update(tpr)
    combined.update(roc_auc)

    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in combined.items()]))

    return df

def make_prediction(model_filepath, X_test, y_test, gene_list):

    """Load model, make predictions and generate classification report"""

    print('\nGenerating classification report...')
    
    pred_model = tf.keras.models.load_model(model_filepath) # Load model
    
    y_pred = pred_model.predict(X_test, batch_size=64, verbose=1) # Make predictions on test set
    
    df_auc = compute_auc(y_pred = y_pred, y_test = y_test, gene_list = gene_list) # Compute AUC
    
    y_pred_rounded = (y_pred == y_pred.max(axis=1, keepdims=1)).astype(float) # Turn into 0s and 1s
    
    report = classification_report(y_test, y_pred_rounded, output_dict=True) # Compute classification report
    
    df_report = pd.DataFrame(report).transpose()
    
    conf_mat_df = generate_confusion_matrix(y_test = y_test, y_pred = y_pred, gene_list = gene_list) # Generate confusion matrix plot

    return df_report, df_auc, conf_mat_df

def train(gene_list, epochs, channel, seed, limit):
    
    """Train CNN on dataset of two genes (usually gene_1 is control)"""
    
    # Generate datasets
    X_train, y_train, X_val, y_val, X_test, y_test = generate_data(gene_list = gene_list, channel = channel, seed = seed, limit = limit)
    
    mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=3))

    with mirrored_strategy.scope():
        model = create_model(classes = len(gene_list))

    model_dir = 'D:\\TMC\\Models'
    report_dir = 'D:\\TMC\\Reports' 
    auc_dir ='D:\\TMC\\AUC'
    history_dir = 'D:\\TMC\\Histories'
    conf_mat_dir = 'D:\\TMC\\Conf_mat_raw'
    gene_names = '_'.join(gene_list)

    if not os.path.exists(os.path.join(model_dir, gene_names)):
        os.mkdir(os.path.join(model_dir, gene_names))

    if not os.path.exists(os.path.join(report_dir, gene_names)):
        os.mkdir(os.path.join(report_dir, gene_names))

    if not os.path.exists(os.path.join(history_dir, gene_names)):
        os.mkdir(os.path.join(history_dir, gene_names))

    if not os.path.exists(os.path.join(auc_dir, gene_names)):
        os.mkdir(os.path.join(auc_dir, gene_names))

    if not os.path.exists(os.path.join(conf_mat_dir, gene_names)):
        os.mkdir(os.path.join(conf_mat_dir, gene_names))
    
    filepath_loss = os.path.join(model_dir, f"{gene_names}\\{gene_names}_{channel}_{seed}.hdf5")
    val_loss_checkpoint = ModelCheckpoint(filepath_loss, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
    
    # Train model
    history = model.fit(X_train, y_train, epochs = epochs,
                        validation_data = (X_test, y_test), batch_size = 512,
                        callbacks = [val_loss_checkpoint])

    # Generate classification report on test set using LOSS model
    df_report_loss, df_auc_loss, conf_mat_df = make_prediction(model_filepath = filepath_loss, X_test = X_test, y_test = y_test, gene_list = gene_list)
    df_report_loss.to_csv(os.path.join(report_dir, f'{gene_names}\\{gene_names}_{channel}_{seed}_report.csv'))
    df_auc_loss.to_csv(os.path.join(auc_dir, f'{gene_names}\\{gene_names}_{channel}_{seed}_AUC.csv'))
    conf_mat_df.to_csv(f'D:\\TMC\\Conf_mat_raw\\{gene_names}\\{gene_names}_{channel}_{seed}_conf_mat.csv')    
    
    df_history = pd.DataFrame.from_dict(history.history)
    df_history.to_csv(os.path.join(history_dir, f'{gene_names}\\{gene_names}_{channel}_{seed}_history.csv'))

def train_all_channels(gene_list, epochs, limit, seed):
    
    """Train a gene against controls for all channels"""
    
    channels = ['AGP', 'ER', 'Mito', 'Nucleus', 'Syto']
    
    for channel in channels:
        train(gene_list = gene_list, epochs = epochs, channel = channel, seed = seed, limit = limit)
        tf.keras.backend.clear_session()