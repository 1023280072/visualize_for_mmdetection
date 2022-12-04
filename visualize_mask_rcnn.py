# 用于mmdetection的yolov3模型输出json文件的可视化
# 如果是其他模型，或许只更改self.loss_name变量就可使用？（简单猜测，作者并没有去尝试）

import glob
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class visualize_mmdetection():
    def __init__(self, path):
        self.path = path
        self.dict_list = list()
        #绘制loss
        self.loss_name = ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'acc', 'loss_bbox', 'loss_mask', 'loss']
        self.train_loss_list = [list() for _ in range(len(self.loss_name))]
        self.val_loss_list = [list() for _ in range(len(self.loss_name))]
        self.epoch1 = list()
        #绘制map
        self.map_name = ['segm_mAP', 'segm_mAP_50']
        self.map_list = [list() for _ in range(len(self.map_name))]
        self.epoch2 = list()

    def load_data(self):
        for line in open(self.path):
            info = json.loads(line)
            self.dict_list.append(info)
        for i in range(1, len(self.dict_list)):
            mode = dict(self.dict_list[i])['mode']
            if mode == 'train':
                for j in range(len(self.loss_name)):
                    tmp_value = dict(self.dict_list[i])[self.loss_name[j]]
                    self.train_loss_list[j].append(tmp_value)
            elif mode == 'val':
                if self.map_name[0] in dict(self.dict_list[i]).keys():
                    for j in range(len(self.map_name)):
                        tmp_value = dict(self.dict_list[i])[self.map_name[j]]
                        self.map_list[j].append(tmp_value)
                    epoch_value = dict(self.dict_list[i])['epoch']
                    self.epoch2.append(epoch_value)
                elif self.loss_name[0] in dict(self.dict_list[i]).keys():
                    for j in range(len(self.loss_name)):
                        tmp_value = dict(self.dict_list[i])[self.loss_name[j]]
                        self.val_loss_list[j].append(tmp_value)
                    epoch_value = dict(self.dict_list[i])['epoch']
                    self.epoch1.append(epoch_value)

    def plot_loss(self):
        plt.rcParams.update({'font.size': 15})
        plt.figure(figsize=(40, 30))
        #train
        i = 0
        plt.subplot(4, 4, 1, title='train_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.train_loss_list[i], 'b')
        i = 1
        plt.subplot(4, 4, 5, title='train_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.train_loss_list[i], 'b')
        i = 2
        plt.subplot(4, 4, 9, title='train_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.train_loss_list[i], 'b')
        i = 3
        plt.subplot(4, 4, 13, title='train_' + self.loss_name[i], ylabel='accuracy')
        plt.plot(self.train_loss_list[i], 'b')
        i = 4
        plt.subplot(4, 4, 3, title='train_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.train_loss_list[i], 'b')
        i = 5
        plt.subplot(4, 4, 7, title='train_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.train_loss_list[i], 'b')
        i = 6
        plt.subplot(4, 4, 11, title='train_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.train_loss_list[i], 'b')
        #val
        i = 0
        plt.subplot(4, 4, 2, title='val_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.epoch1, self.val_loss_list[i], 'g')
        i = 1
        plt.subplot(4, 4, 6, title='val_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.epoch1, self.val_loss_list[i], 'g')
        i = 2
        plt.subplot(4, 4, 10, title='val_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.epoch1, self.val_loss_list[i], 'g')
        i = 3
        plt.subplot(4, 4, 14, title='val_' + self.loss_name[i], ylabel='accuracy')
        plt.plot(self.epoch1, self.val_loss_list[i], 'g')
        i = 4
        plt.subplot(4, 4, 4, title='val_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.epoch1, self.val_loss_list[i], 'g')
        i = 5
        plt.subplot(4, 4, 8, title='val_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.epoch1, self.val_loss_list[i], 'g')
        i = 6
        plt.subplot(4, 4, 12, title='val_' + self.loss_name[i], ylabel='loss')
        plt.plot(self.epoch1, self.val_loss_list[i], 'g')
        plt.savefig((self.path + '_loss.png'))

    def plot_map(self):
        plt.rcParams.update({'font.size': 15})
        plt.figure(figsize=(20, 10))
        i = 0
        plt.subplot(121, title=self.map_name[i], xlabel='epoch', ylabel='map')
        plt.plot(self.epoch2, self.map_list[i], 'b')
        max_map = max(self.map_list[i])
        max_map_idx = self.map_list[i].index(max_map)
        max_map_epoch = self.epoch2[max_map_idx]
        plt.plot(max_map_epoch, max_map, 'o')
        plt.text(max_map_epoch, max_map, '(%d,%.3f)' % (max_map_epoch, max_map), fontsize=15, color='r')
        i += 1
        plt.subplot(122, title=self.map_name[i], xlabel='epoch', ylabel='map')
        plt.plot(self.epoch2, self.map_list[i], 'b')
        max_map = max(self.map_list[i])
        max_map_idx = self.map_list[i].index(max_map)
        max_map_epoch = self.epoch2[max_map_idx]
        plt.plot(max_map_epoch, max_map, 'o')
        plt.text(max_map_epoch, max_map, '(%d,%.3f)' % (max_map_epoch, max_map), fontsize=15, color='r')
        plt.savefig((self.path + '_map.png'))

if __name__ == '__main__':
    jsons_path = './work_dirs/mask_rcnn_r50_fpn_1x_coco/*.json'
    json_path =  glob.glob(jsons_path)[0]
    x = visualize_mmdetection(json_path)
    x.load_data()
    x.plot_loss()
    x.plot_map()