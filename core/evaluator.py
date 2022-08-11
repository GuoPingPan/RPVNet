import torch
import torch.distributed as dist

class MeanIoU(object):
    '''
        meanIOU: \frac{1}{C}\sum^C_{c=1} \frac{TP}{TP+FP+FN}

    '''
    def __init__(self,num_classes,rank,ignore_label=None):
        super(MeanIoU, self).__init__()

        self.num_classes = num_classes
        self.rank = rank
        self.total_target = torch.zeros(num_classes) # TP+FN
        self.total_pred = torch.zeros(num_classes) # TP+FP
        self.total_correct = torch.zeros(num_classes) # TP
        self.ignore_label = ignore_label

    def epoch_miou(self):

        with torch.no_grad():
            if self.rank == 0:
                for i in range(self.num_classes):
                    self.total_target[i] = dist.all_reduce(self.total_target[i])
                    self.total_pred[i] = dist.all_reduce(self.total_pred[i])
                    self.total_correct[i] = dist.all_reduce(self.total_correct[i])

            iou = self.total_correct / (self.total_pred + self.total_target - self.total_correct + 0.001)
            miou = torch.mean(iou)
            acc = torch.mean(self.total_correct / self.total_pred + 0.001)
            # print("correct",self.total_correct)
            # print("pred", self.total_pred)
        return iou,miou,acc

    def reset(self):
        self.total_target = torch.zeros(self.num_classes) # TP+FN
        self.total_pred = torch.zeros(self.num_classes) # TP+FP
        self.total_correct = torch.zeros(self.num_classes) # TP


    def __call__(self,preds,targets):

        # print(preds.shape,targets.shape)

        pre = preds[targets!=self.ignore_label]
        tar = targets[targets!=self.ignore_label]

        batch_target = torch.zeros(self.num_classes)
        batch_pred = torch.zeros(self.num_classes)
        batch_correct = torch.zeros(self.num_classes)

        for i in range(self.num_classes):
            # print(temp1,temp2,temp3)
            batch_target[i] = torch.sum(tar==i).item()
            batch_pred[i] = torch.sum(pre==i).item()
            batch_correct[i] = torch.sum((tar==i) & (pre==tar)).item()

            self.total_target[i] += batch_target[i]
            self.total_pred[i] += batch_pred[i]
            self.total_correct[i] += batch_correct[i]

        iou = batch_correct /(batch_pred+batch_target-batch_correct+0.001)
        miou = torch.mean(iou)
        acc = torch.mean(batch_correct/(batch_pred+0.001))

        return iou,miou,acc

