import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os.path
import sys
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from core.dataset.semantic_kitti import SemanticKITTIInternal
from core.models.rpvnet import RPVnet
from core.evaluator import MeanIoU

batch_size = None
device = None

def main(args):

    assert os.path.exists(args.dataset),(f'The dataset dir [{args.dataset}] doesn\'t exist.')
    assert os.path.exists(args.model_cfg),(f'The model config [{args.model_cfg}] doesn\'t exist.')
    assert os.path.exists(args.data_cfg),(f'The dataset config [{args.data_cfg}] doesn\'t exist.')


    print('-----------------')
    print(f'dataset dir: {args.dataset}')
    print(f'model config: {args.model_cfg}')
    print(f'dataset config: {args.data_cfg}')
    print(f'checkpoint: {args.checkpoint}')
    print('-----------------')

    # load dataset config
    try:
        data_cfg = yaml.safe_load(open(args.data_cfg,'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # load model config
    try:
        model_cfg = yaml.safe_load(open(args.model_cfg,'r'))
    except Exception as e:
        print(e)
        print("Error opening model yaml file.")
        quit()

    # check if use pretrained model
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint) and args.checkpoint.endswith('.ckpt'):
            print(f'Using the pretrained model:[{args.checkpoint}]')
        else:
            print(f'The pretrained model:[{args.checkpoint}] doesn\'t exist.')

    model = RPVnet(
        vsize=model_cfg['voxel_size'],
        cr=model_cfg['cr'],
        cs=model_cfg['dimension_of_stages'],
        num_classes=model_cfg['num_classes']
    )

    data = SemanticKITTIInternal(
        root=args.dataset,
        voxel_size=data_cfg['voxel_size'],
        range_size=data_cfg['range_size'],
        sample_stride=data_cfg['sample_stride'],
        split=data_cfg['split']['test'],
        max_voxels=data_cfg['max_voxels'],
        label_name_mapping=data_cfg['label_name_mapping'],
        kept_labels=data_cfg['kept_labels']
    )

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=args.device)
        print(state['info'])
        args.epochs = state['info']['epochs']
        model.load_state_dict(state['state_dict'])

        criterion = getattr(nn,model_cfg['train']['loss'])(ignore_index=model_cfg['train']['ignore_index'])
        evaluator = MeanIoU(num_classes=model_cfg['num_classes'],rank=-100,ignore_label=model_cfg['train']['ignore_index'])
        batch_size = model_cfg['train']['batch_size']
        device = args.device
        # num of worker
        nw = min([os.cpu_count(), model_cfg['train']['batch_size'], 8])

        if args.device == 'cpu':
            dataloader = DataLoader(data,
                                         batch_size= batch_size,
                                         num_workers=nw,
                                         collate_fn=data.collate_fn,
                                         shuffle=True)

        elif torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True
            # 单卡
            if torch.cuda.device_count() == 1:
                dataloader = DataLoader(data,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             collate_fn=data.collate_fn,
                                             shuffle=True)
                model.to(device)

    for epoch in range(args.epochs):

        loss, (iou, miou, acc) = test_each_epoch(epoch,dataloader,model,evaluator,criterion)
        print(f'Epoch:[{epoch + 1:>3d}/{args.epochs:>3d}]'
              f'   Mean Loss:{loss}   mIoU:{miou}   Accuary:{acc}')


def test_each_epoch(epoch,dataloader,model,evaluator,criterion):
    dataloader = tqdm(dataloader, file=sys.stdout)

    torch.cuda.empty_cache()
    model.eval()
    evaluator.reset()

    mean_loss = torch.zeros(1).to(device)

    for batch, data in enumerate(dataloader):
        lidar, label, image = data['lidar'], data['label'], data['image']
        py, px = data['py'], data['px']

        if device == 'cuda':
            lidar, label, image = lidar.cuda(), label.cuda(), image.cuda()
            px = [x.cuda() for x in px]
            py = [y.cuda() for y in py]

        outputs = model(lidar, image, py, px)

        loss = criterion(outputs, label.F.long())
        loss.backward()

        mean_loss = (mean_loss * batch + loss.detach()) / (batch + 1)
        iou, miou, acc = evaluator(outputs.argmax(dim=1), label.F.long())

        assert torch.isfinite(loss), f'ERROR: non-finite loss, ending training! {loss}'

        print(f'Batch:[{batch + 1:>3d}/{batch_size}]'
              f'   Mean Loss:{mean_loss}   mIoU:{miou}   Accuary:{acc}')


    return mean_loss, evaluator.epoch_miou()


if __name__=='__main__':

    root,_ = os.path.split(os.path.abspath(__file__))

    parser = argparse.ArgumentParser('Trainning Model.')
    parser.add_argument(
        '--dataset','-d',
        type=str,required=True,
        help='the root dir of datasets'
    )
    parser.add_argument(
        '--checkpoint','-ckpt',
        type=str,
        default=None,
        help='the path for loading checkpoint'
    )
    parser.add_argument(
        '--model_cfg','-m',
        type=str,
        default=root + '/config/model.yaml',
        help='the config of model'
    )
    parser.add_argument(
        '--data_cfg','-dc',
        type=str,
        default=root + '/config/semantic-kitti.yaml',
        help='the config of model'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='device id (i.e. 0 or 0,1 or cuda)'
    )

    args,unparsed = parser.parse_known_args()

    main(args)

























