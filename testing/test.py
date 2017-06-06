import cv2 as cv 
import numpy as np
import scipy
import PIL.Image
import math
import sys
sys.path.append("/home/robbie/caffe_train/python")
import caffe
import time
from config_reader import config_reader
import util
import copy
import matplotlib
import pylab as plt
import sys, os, random

num_test = 100
num_models = 4

#generate test images
test_images = [None] * num_test
for i in xrange(num_test):
    test_image = random.choice(os.listdir('/home/robbie/data/rmppe/training/dataset/COCO/images/val2014'))
    test_image = os.path.join('/home/robbie/data/rmppe/training/dataset/COCO/images/val2014', test_image)
    test_images[i] = cv.imread(test_image) # B,G,R order
caffe.set_mode_gpu()
caffe.set_device(0)

#lists to store averages computed by original model, which we treat as ground truth
true_heatmap_avg = [None] * num_test
true_paf_avg = [None] * num_test

for i in xrange(1): #0 is original model
    #we take the Euclidean distance between the average heatmap/paf computed by the smaller network vs the original as the loss
    heatmap_loss = 0
    paf_loss = 0
    times = np.empty((num_test, 4))
    param, model = config_reader(4)
    net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)
    for j in xrange(num_test):
        oriImg = test_images[j]
        multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        for m in xrange(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])
            
            net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
            net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
            
            start_time = time.time()
            output_blobs = net.forward()
            total_time = (time.time() - start_time) * 1000
            times[j][m] = total_time
            
            # extract outputs, resize, and remove padding
            if output_blobs.keys()[1].endswith("L2"): #L2 predicts heatmaps, L1 predicts PAFs
                heatmap_index = 1
                paf_index = 0
            else:
                heatmap_index = 0
                paf_index = 1
            heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[heatmap_index]].data), (1,2,0))
            heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

            paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[paf_index]].data), (1,2,0))
            paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
            paf_avg = paf_avg + paf / len(multiplier)
        if i == 0: #original model, we just need to store computed averages
            true_heatmap_avg[j] = heatmap_avg
            true_paf_avg[j] = paf_avg
        else: #compute loss
            heatmap_loss += np.linalg.norm(true_heatmap_avg[j] - heatmap_avg)
            paf_loss += np.linalg.norm(true_paf_avg[j] - paf_avg)
    times = times.mean(axis=0)
    print model['description']
    for m in xrange(4):
        print('At scale %d, The CNN took an average of %.2f ms.' % (m, times[m]))
    heatmap_loss /= num_test
    paf_loss /= num_test
    print('Average heatmap loss: %f\nAverage PAF loss: %f' % (heatmap_loss, paf_loss))
