import torch
import torch.distributed as dist

class MeanIoU(object):
    '''
        meanIOU: \frac{1}{C}\sum^C_{c=1} \frac{TP}{TP+FP+FN}

    '''
    def __init__(self,num_classes,ignore_label=None):
        super(MeanIoU, self).__init__()

        self.num_classes = num_classes
        self.total_target = torch.zeros(num_classes) # TP+FN
        self.total_pred = torch.zeros(num_classes) # TP+FP
        self.total_corect = torch.zeros(num_classes) # TP
        self.ignore_label = self.ignore_label

    def epoch_miou(self):

        for i in range(self.num_classes):
            self.total_target[i] = dist.all_reduce(self.total_target[i],reduction='sum')
            self.total_pred[i] = dist.all_reduce(self.total_pred[i],reduction='sum')
            self.total_corect[i] = dist.all_reduce(self.total_corect[i],reduction='sum')

        iou = self.total_corect /(self.total_pred+self.total_target-self.total_corect)
        miou = torch.mean(iou)
        acc = torch.mean(self.total_corect/self.total_pred)
        return iou,miou,acc

    def reset(self):
        self.total_target = torch.zeros(num_classes) # TP+FN
        self.total_pred = torch.zeros(num_classes) # TP+FP
        self.total_corect = torch.zeros(num_classes) # TP


    def __call__(self,preds,targets):

        pre = preds[targets!=self.ignore_label]
        tar = targets[targets!=self.ignore_label]
        batch_target = torch.zeros(self.num_classes)
        batch_pred = torch.zeros(self.num_classes)
        batch_correct = torch.zeros(self.num_classes)

        for i in range(self.num_classes):
            temp1 = torch.sum(tar==i).item()
            temp2 = torch.sum(tar==i).item()
            temp3 = torch.sum((tar==i) and (pre==tar)).item()
            batch_target[i]= temp1
            batch_pred[i] = temp2
            batch_correct[i] += temp3

            self.total_target[i] += temp1
            self.total_pred[i] += temp2
            self.total_corect[i] += temp3

        iou = batch_correct /(batch_pred+batch_target-batch_pred)
        miou = torch.mean(iou)
        acc = torch.mean(batch_correct/batch_pred)

        return iou,miou,acc

