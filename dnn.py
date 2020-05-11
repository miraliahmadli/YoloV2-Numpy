import os
import sys
import math
import networkx as nx
import numpy as np

import multiprocessing

class DnnInferenceEngine(object):
    def __init__(self, graph):
        self.g = graph

    def run(self, tin):
        self.g.in_node.set_input(tin)
        out = {}
        currents = [self.g.in_node]
        done = set()
        while (len(currents) != 0):
            nexts = []
            for current in currents:
                skip_current = False
                predecessors = self.g.G.predecessors(current)
                for predecessor in predecessors:
                    if predecessor not in done:
                        nexts.append(predecessor)
                        skip_current = True
                if skip_current:
                    continue
                current.run()
                if self.g.is_out_node(current):
                    out = current.result
                done.add(current)
                for successor in self.g.G.successors(current):
                    nexts.append(successor)
            currents = nexts
        return out

class DnnGraphBuilder(object):
    def __init__(self):
        self.G = nx.DiGraph()
        self.name_num = {"conv2d": 0, 
                         "bias_add": 0, 
                         "max_pool2d": 0, 
                         "batch_norm": 0, 
                         "leaky_relu": 0, 
                         "input": 0}
        self.in_node = None
        self.out_node = None

    def set_in_node(self, node):
        self.in_node = node

    def set_out_node(self, node):
        self.out_node = node

    def is_out_node(self, node):
        return self.out_node is node

    def get_name(self, layer_name):
        name = layer_name + "_" + str(self.name_num[layer_name])
        self.name_num[layer_name] += 1
        return name

    def create_conv2d(self, in_node, kernel, strides, padding):
        out_node = Conv2D(self.get_name("conv2d"), in_node, kernel, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_bias_add(self, in_node, biases):
        out_node = BiasAdd(self.get_name("bias_add"), in_node, biases)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_max_pool2d(self, in_node, ksize, strides, padding):
        out_node = MaxPool2D(self.get_name("max_pool2d"), in_node, ksize, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_batch_norm(self, in_node, mean, variance, gamma, epsilon):
        out_node = BatchNorm(self.get_name("batch_norm"), in_node, mean, variance, gamma, epsilon)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_leaky_relu(self, in_node):
        out_node = LeakyReLU(self.get_name("leaky_relu"), in_node)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_input(self, in_shape):
        out_node = Input(self.get_name("input"), in_shape)
        self.G.add_node(out_node) 
        self.set_in_node(out_node)  # Assume there's only one input
        return out_node

class DnnNode(object):
    def __init__(self):
        pass

    def run(self):
        self.result = None 

#
# Complete below classes.
#

class Conv2D(DnnNode):
    def __init__(self, name, in_node, kernel, strides, padding):
        self.in_node = in_node
        self.batch, self.in_height, self.in_width, self.in_channels = in_node.out_shape
        self.filter_height, self.filter_width, self.kernel_in_channels, self.out_channels = kernel.shape

        assert self.in_channels == self.kernel_in_channels, \
        "Shape of filters must be same to number of input channels, %d is not equal to %d" % (self.kernel_in_channels, self.in_channels)

        assert padding in ["SAME", "VALID"], \
        "Invalid padding name: %s, should be either SAME or VALID" % padding

        self.name = name
        print(self.name)
        self.kernel = kernel
        self.padding = (padding == "SAME")
        self.strides = strides

        self.pad_h = 0
        self.pad_w = 0
        if self.padding:
            # ((s-1) * x + k -s)/ 2
            self.pad_h = ((self.strides[1] - 1) * self.in_height + self.filter_height - self.strides[1]) // 2
            self.pad_w = ((self.strides[2] - 1) * self.in_width + self.filter_width - self.strides[2]) // 2
        
        self.output_height  = int((self.in_height - self.filter_height + 2 * self.pad_h) / self.strides[1] + 1)
        self.output_width   = int((self.in_width - self.filter_width + 2 * self.pad_w) / self.strides[2] + 1)
        self.result = np.zeros((self.batch, self.output_height, self.output_width, self.out_channels))
        self.out_shape = self.result.shape

    def run(self):
        def multi(b, i, j, k):
            for h in range(self.filter_height):
                for w in range(self.filter_width):
                    for q in range(self.in_channels):
                        self.result[b, i, j, k] = \
                            (self.in_node.result[b, self.strides[1] * i + h, self.strides[2] * j + w, q] * 
                                self.kernel[h, w, q, k])
            # average
            self.result[b, i, j, k] /= (self.filter_width * self.filter_height * self.in_channels)


        for b in range(self.batch):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    for k in range(self.out_channels):
                        # can be multiprocessed
                        for h in range(self.filter_height):
                            for w in range(self.filter_width):
                                for q in range(self.in_channels):
                                    self.result[b, i, j, k] = \
                                        (self.in_node.result[b, self.strides[1] * i + h, self.strides[2] * j + w, q] * 
                                            self.kernel[h, w, q, k])
                        
                        # average
                        self.result[b, i, j, k] /= (self.filter_width * self.filter_height * self.in_channels)

        

        # output[b, i, j, k] =
        #     sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q]
        #                     * filter[di, dj, q, k]
class BiasAdd(DnnNode):
    def __init__(self, name, in_node, biases):
        self.in_node = in_node
        self.prev_res =  self.in_node.result
        self.batch, self.in_height, self.in_width, self.in_channels = in_node.out_shape
        assert self.in_channels == biases.shape[0], \
        "Shape of biases must be equal to number of input channels, %d is not equal to %d" % (biases.shape[0], self.in_channels)

        self.biases = biases
        self.name = name
        print(self.name)
        self.result = np.zeros(self.in_node.out_shape)
        self.out_shape = self.result.shape

    def run(self):
        #self.result = np.zeros(self.in_node.out_shape) #self.in_node.result[:,:,:,:] #np.zeros(self.in_node.out_shape)
        def multi(b, h, w, c):
            self.result[b, h, w, c] = self.prev_res[b, h, w, c] + self.biases[c]
        
        for b in range(self.batch):
            for h in range(self.in_height):
                for w in range(self.in_width):
                    for c in range(self.in_channels):
                        self.result[b, h, w, c] = self.prev_res[b, h, w, c] + self.biases[c]

class MaxPool2D(DnnNode):
    def __init__(self, name, in_node, ksize, strides, padding):
        self.batch, self.in_height, self.in_width, self.in_channels = in_node.out_shape
        self.out_channels = self.in_channels
        assert padding in ["SAME", "VALID"], \
        "Invalid padding name: %s, should be either SAME or VALID" % padding

        self.strides = strides
        self.ksize = ksize
        self.name = name
        print(self.name)
        self.in_node = in_node
        
        self.padding = (padding == "SAME")

        self.pad_h = 0
        self.pad_w = 0
        if self.padding:
            # ((s-1) * x + k -s)/ 2
            self.pad_h = ((self.strides[1] - 1) * self.in_height + self.ksize[1] - self.strides[1]) // 2
            self.pad_w = ((self.strides[2] - 1) * self.in_width + self.ksize[2] - self.strides[2]) // 2
            self.prev_res = np.zeros((self.batch, self.in_height + 2*self.pad_h, self.in_width + 2*self.pad_w, self.in_channels))
            prev_copy = np.copy(self.in_node.result)
            self.prev_res[:, self.pad_h : -self.pad_h, self.pad_w : -self.pad_w, :] = prev_copy
        else:
            self.prev_res = np.copy(self.in_node.result)
        
        self.output_height  = int((self.in_height - self.ksize[1] + 2 * self.pad_h) / self.strides[1] + 1)
        self.output_width   = int((self.in_width - self.ksize[2] + 2 * self.pad_w) / self.strides[2] + 1)
        self.result = np.zeros((self.batch, self.output_height, self.output_width, self.out_channels))
        self.out_shape = self.result.shape
        
    def run(self):
        def get_max_from_window(b, x, y, c):
            # window = self.prev_res[ b, x : min(x + self.ksize, self.in_height), 
            #                         y : min(y + self.ksize, self.in_width), c]
            max_num = float("-inf")
            for i in range(x, min(x + self.ksize, self.in_height + 2*self.pad_h)):
                for j in range(y : min(y + self.ksize, self.in_width + 2*self.pad_w)):
                    if self.prev_res[b, i, j, c] > max_num:
                        max_num = self.prev_res[b, i, j, c]
            return max_num

        def multi(b, i, j, c):
            # m = get_max_from_window(b, i, j, c)
            self.result[b, i, j, c] = get_max_from_window(b, i, j, c)

        for b in range(self.batch):
            for i in range(0, self.output_height, self.strides[1]):
                for j in range(0, self.output_width, self.strides[2]):
                    for c in range(self.out_channels):
                        m = get_max_from_window(b, i, j, c)
                        self.result[b, i, j, c] = m


class BatchNorm(DnnNode):
    def __init__(self, name, in_node, mean, variance, gamma, epsilon):
        self.in_node = in_node
        self.batch, self.in_height, self.in_width, self.in_channels = in_node.out_shape

        assert self.in_channels == mean.shape[0], \
        "Shape of mean must be equal to number of input channels, %d is not equal to %d" % (mean.shape[0], self.in_channels)

        assert self.in_channels == variance.shape[0], \
        "Shape of variance must be equal to number of input channels, %d is not equal to %d" % (variance.shape[0], self.in_channels)

        assert self.in_channels == gamma.shape[0], \
        "Shape of gamma must be equal to number of input channels, %d is not equal to %d" % (gamma.shape[0], self.in_channels)

        # assert self.in_channels == epsilon.shape[0], \
        # f"Shape of epsilon must be equal to number of input channels, {epsilon.shape[0]} is not equal to {self.in_channels}"
        
        self.name = name
        print(self.name)

        self.mean = mean
        self.epsilon = epsilon
        self.gamma = gamma
        self.variance = variance
        self.result = np.copy(self.in_node.result)
        self.out_shape = self.result.shape

    def run(self):
        # self.result = np.copy(self.in_node.result)#np.zeros(self.in_node.out_shape)
        def multi(b, h, w, c):
            self.result[b, h, w, c] = \
                (self.result[b, h, w, c] - self.mean[c]) / np.sqrt(self.variance[c] + self.epsilon)
        for b in range(self.batch):
            for h in range(self.in_height):
                for w in range(self.in_width):
                    for c in range(self.in_channels):
                        self.result[b, h, w, c] = \
                            (self.result[b, h, w, c] - self.mean[c]) / np.sqrt(self.variance[c] + self.epsilon)

class LeakyReLU(DnnNode):
    def __init__(self, name, in_node):
        self.in_node = in_node
        self.batch, self.in_height, self.in_width, self.in_channels = in_node.out_shape
        self.alpha = 0.01
        self.name = name
        print(self.name)
        self.result = np.copy(self.in_node.result)
        self.out_shape = self.result.shape

    def run(self):
        # self.result[self.result < 0] *= self.alpha
        def multi(b, h, w, c):
            if self.result[b, h, w, c] < 0:
                self.result[b, h, w, c] *= self.alpha
        for b in range(self.batch):
            for h in range(self.in_height):
                for w in range(self.in_width):
                    for c in range(self.in_channels):
                        if self.result[b, h, w, c] < 0:
                            self.result[b, h, w, c] *= self.alpha


# Do not modify below
class Input(DnnNode):
    def __init__(self, name, in_shape):
        self.name = name
        # print(self.name)
        self.in_shape = in_shape 
        self.out_shape =in_shape
        self.result = np.ndarray(self.in_shape)

    def set_input(self, tensor):
        assert tuple(self.in_shape) == tuple(tensor.shape)
        self.result = tensor 

    def run(self):
        pass

