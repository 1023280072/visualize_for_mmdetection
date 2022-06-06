# 用于mmdetection的yolov3模型输出json文件的可视化
# 如果是其他模型，或许只更改self.loss_name变量就可使用？（简单猜测，作者并没有去尝试）

import sys
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class visualize_mmdetection():
    def __init__(self, path):
        self.path = path
        self.dict_list = list()
        #绘制loss
        self.loss_name = ['loss_wh', 'loss_xy', 'loss_conf', 'loss_cls', 'loss']
        self.loss_list = [list() for _ in range(len(self.loss_name))]
        #绘制map
        self.map_name = 'bbox_mAP_50'
        self.map_list = list()
        self.epoch = list()

    def load_data(self):
        for line in open(self.path):
            info = json.loads(line)
            self.dict_list.append(info)
        for i in range(1, len(self.dict_list)):
            mode = dict(self.dict_list[i])['mode']
            if mode == 'train':
                for j in range(len(self.loss_name)):
                    tmp_value = dict(self.dict_list[i])[self.loss_name[j]]
                    self.loss_list[j].append(tmp_value)
            elif mode == 'val':
                if self.map_name in dict(self.dict_list[i]).keys():
                    tmp_value = dict(self.dict_list[i])[self.map_name]
                    epoch_value = dict(self.dict_list[i])['epoch']
                    self.map_list.append(tmp_value)
                    self.epoch.append(epoch_value)

    def plot_loss(self):
        plt.rcParams.update({'font.size': 15})
        plt.figure(figsize=(20, 20))
        i = 0
        plt.subplot(321, title=self.loss_name[i], ylabel='loss')
        plt.plot(self.loss_list[i])
        i += 1
        plt.subplot(322, title=self.loss_name[i], ylabel='loss')
        plt.plot(self.loss_list[i])
        i += 1
        plt.subplot(323, title=self.loss_name[i], ylabel='loss')
        plt.plot(self.loss_list[i])
        i += 1
        plt.subplot(324, title=self.loss_name[i], ylabel='loss')
        plt.plot(self.loss_list[i])
        i += 1
        plt.subplot(325, title=self.loss_name[i], ylabel='loss')
        plt.plot(self.loss_list[i])
        plt.savefig((sys.argv[1] + '_loss.png'))

    def plot_map(self):
        plt.rcParams.update({'font.size': 15})
        plt.figure(figsize=(20, 20))
        plt.plot(self.epoch, self.map_list)
        plt.title(self.map_name)
        plt.xlabel('epoch')
        plt.ylabel('map')
        plt.savefig((sys.argv[1] + '_map.png'))

if __name__ == '__main__':
    x = visualize_mmdetection(sys.argv[1])
    x.load_data()
    x.plot_loss()
    x.plot_map()