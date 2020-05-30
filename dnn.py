import os
import sys
import math
import networkx as nx
import numpy as np

import multiprocessing
import itertools
# from multiprocessing import set_start_method
num_cpu = multiprocessing.cpu_count()

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
            currents = nexts[:]
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
        batch, in_height, in_width, in_channels = in_node.out_shape
        filter_height, filter_width, kernel_in_channels, out_channels = kernel.shape

        assert in_channels == kernel_in_channels, \
        "Shape of filters must be same to number of input channels, %d is not equal to %d" % (kernel_in_channels, in_channels)

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
            # to avoid  checking extra cases, we will not divide by two
            self.pad_h = ((self.strides[1] - 1) * in_height + filter_height - self.strides[1])
            self.pad_w = ((self.strides[2] - 1) * in_width + filter_width - self.strides[2])
        output_height  = int(((in_height - filter_height + self.pad_h) / self.strides[1]) + 1)
        output_width   = int(((in_width - filter_width + self.pad_w) / self.strides[2]) + 1)
        self.prev_res = np.zeros((batch, in_height + self.pad_h, in_width + self.pad_w, in_channels))
        self.result = np.zeros((batch, output_height, output_width, out_channels))
        self.out_shape = self.result.shape
        self.filter_height, self.filter_width, self.in_channels, out_channels = kernel.shape

    def run(self):
        filter_height, filter_width, in_channels, out_channels = self.kernel.shape
        batch, in_height, in_width, in_channels = self.in_node.out_shape
        if self.pad_h == 0 and self.pad_w == 0:
            self.prev_res = self.in_node.result
        else:
            self.prev_res[:, self.pad_h//2 : -((self.pad_h+1)//2), self.pad_w//2 : -((self.pad_w + 1)//2), :] = self.in_node.result
        batch, output_height, output_width, out_channels = self.out_shape

        for b in range(batch):
            for i in range(output_height):
                for j in range(output_width):
                    for k in range(out_channels):
                        # can be multiprocessed
                        for h in range(filter_height):
                            for w in range(filter_width):
                                for q in range(in_channels):
                                    self.result[b, i, j, k] += \
                                        (self.prev_res[b, self.strides[1] * i + h, self.strides[2] * j + w, q] * 
                                            self.kernel[h, w, q, k])

class BiasAdd(DnnNode):
    def __init__(self, name, in_node, biases):
        self.in_node = in_node
        batch, in_height, in_width, in_channels = in_node.out_shape
        assert in_channels == biases.shape[0], \
        "Shape of biases must be equal to number of input channels, %d is not equal to %d" % (biases.shape[0], in_channels)

        self.biases = biases
        self.name = name
        print(self.name)
        self.result = np.zeros(in_node.out_shape)
        self.out_shape = self.result.shape
        self.prev_res= np.zeros(in_node.out_shape)

    def run(self):
        self.prev_res = self.in_node.result
        batch, output_height, output_width, out_channels = self.out_shape
        for b in range(batch):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(out_channels):
                        self.result[b, h, w, :] = self.prev_res[b, h, w, :] + self.biases

class MaxPool2D(DnnNode):
    def __init__(self, name, in_node, ksize, strides, padding):
        self.in_node = in_node
        batch, in_height, in_width, in_channels = in_node.out_shape
        out_channels = in_channels
        assert padding in ["SAME", "VALID"], \
        "Invalid padding name: %s, should be either SAME or VALID" % padding

        self.strides = strides
        self.ksize = ksize
        self.name = name
        print(self.name)
        
        self.padding = (padding == "SAME")

        pad_h = 0
        pad_w = 0
        if self.padding:
            # ((s-1) * x + k -s)/ 2
            pad_h = self.ksize[1] - 1
            pad_w = self.ksize[2] - 1
        self.prev_res = np.zeros((batch, in_height + pad_h, in_width + pad_w, in_channels))
        output_height  = int((in_height - self.ksize[1] + pad_h) / self.strides[1] + 1)
        output_width   = int((in_width - self.ksize[2] + pad_w) / self.strides[2] + 1)
        self.result = np.zeros((batch, output_height, output_width, out_channels))
        self.out_shape = self.result.shape
        
    def run(self):
        batch, in_height, in_width, in_channels = self.in_node.out_shape
        pad_h = 0
        pad_w = 0
        if self.padding:
            pad_h = self.ksize[1] - 1
            pad_w = self.ksize[2] - 1
            self.prev_res[:, pad_h//2 : -((pad_h+1)//2), pad_w//2 : -((pad_w+1)//2), :] = self.in_node.result
        else:
            self.prev_res = self.in_node.result
        batch, in_height, in_width, in_channels = self.prev_res.shape
        batch, output_height, output_width, out_channels = self.out_shape

        for b in range(batch):
            for i in range(output_height):
                for j in range(output_width):
                    for c in range(out_channels):
                        self.result[b, i, j, c] = \
                            np.amax(self.prev_res[b, i * self.strides[1] : i * self.strides[1] + self.ksize[1], 
                                    j * self.strides[2] : j * self.strides[2] + self.ksize[2], c])


class BatchNorm(DnnNode):
    def __init__(self, name, in_node, mean, variance, gamma, epsilon):
        self.in_node = in_node
        batch, in_height, in_width, in_channels = in_node.out_shape

        assert in_channels == mean.shape[0], \
        "Shape of mean must be equal to number of input channels, %d is not equal to %d" % (mean.shape[0], in_channels)

        assert in_channels == variance.shape[0], \
        "Shape of variance must be equal to number of input channels, %d is not equal to %d" % (variance.shape[0], in_channels)

        assert in_channels == gamma.shape[0], \
        "Shape of gamma must be equal to number of input channels, %d is not equal to %d" % (gamma.shape[0], in_channels)
        
        self.name = name
        print(self.name)

        self.mean = mean
        self.epsilon = epsilon
        self.gamma = gamma
        self.variance = variance
        self.result = np.zeros(in_node.out_shape)
        self.out_shape = self.result.shape
        self.prev_res = np.zeros(in_node.out_shape)

    def run(self):
        self.prev_res = self.in_node.result
        batch, output_height, output_width, out_channels = self.out_shape
        std = np.sqrt(self.variance + self.epsilon)
        for b in range(batch):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(out_channels):
                        self.result[b, h, w, c] = self.gamma[c] * (self.prev_res[b, h, w, c] - self.mean[c]) / std[c]

class LeakyReLU(DnnNode):
    def __init__(self, name, in_node):
        self.in_node = in_node
        batch, in_height, in_width, in_channels = in_node.out_shape
        self.alpha = 0.1
        self.name = name
        print(self.name)
        self.result=  np.zeros(in_node.out_shape)
        self.out_shape = self.result.shape

    def run(self):
        self.result = np.copy(self.in_node.result)
        self.result[self.result < 0] *= self.alpha


# Do not modify below
class Input(DnnNode):
    def __init__(self, name, in_shape):
        self.name = name
        # print(self.name)
        self.in_shape = in_shape 
        self.out_shape =in_shape
        self.result = np.zeros(self.in_shape)

    def set_input(self, tensor):
        assert tuple(self.in_shape) == tuple(tensor.shape)
        self.result = tensor 

    def run(self):
        pass
