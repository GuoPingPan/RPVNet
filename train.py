import shutil
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import argparse
import os.path
import datetime

import yaml
from core.trainer import Trainer


def main(args):


    assert os.path.exists(args.dataset),(f'The dataset dir [{args.dataset}] doesn\'t exist.')
    assert os.path.exists(args.model_cfg),(f'The model config [{args.model_cfg}] doesn\'t exist.')
    assert os.path.exists(args.data_cfg),(f'The dataset config [{args.data_cfg}] doesn\'t exist.')


    # root,_ = os.path.split(os.path.abspath(__file__))
    # if args.log is not None:
    #     default= root + '/log/' + \
    #              datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
    #     if os.path.exists(args.log) is None:
    #         print(f'The log dir [{args.dataset}] doesn\'t exist, trying to use'
    #               f'[{default}] as default, please input [y/n] to ensure.')
    #         sig = input()
    #         if sig in ['N','n']: exit()
    #         else: args.log = default

    print('-----------------')
    print(f'dataset dir: {args.dataset}')
    print(f'log dir: {args.log}')
    print(f'model config: {args.model_cfg}')
    print(f'dataset config: {args.data_cfg}')
    print(f'pretrained: {args.checkpoint}')
    print('-----------------')

    try:
        data_cfg = yaml.safe_load(open(args.data_cfg,'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    try:
        model_cfg = yaml.safe_load(open(args.model_cfg,'r'))
    except Exception as e:
        print(e)
        print("Error opening model yaml file.")
        quit()

    if args.log is None:
        root,_ = os.path.split(os.path.abspath(__file__))
        default= root + '/log/' + \
                 datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
        print(f'The log dir [{args.log}] doesn\'t exist, Do you want to use'
              f'[{default}] as default? [y/n]')
        sig = input()
        if sig in ['Y','y']:
            args.log = default
            os.mkdir(root+'/log/')
            os.mkdir(default)
        else:
            print(f'Check the log dir.')
            quit()
    else:
        if os.path.isdir(args.log):
            if os.listdir(args.log):
                print(f'Log dir [{args.log}] is not empty, Do you want to proceed? [y/n]')
                sig = input()
                if sig in ['Y', 'y']:
                    shutil.rmtree(args.log)
                else:
                    print(f'Check the log dir.')
                    quit()
            os.mkdir(args.log)

    # todo
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint) and args.checkpoint.endwith('.ckpt'):
            print(f'Using the pretrained model:[{args.checkpoint}]')


    # arg.device,arg.freeze_layers,args.log,args.pretrained
    train = Trainer(model_cfg,data_cfg,args)
    train.train()




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
        '--log','-l',
        type=str,
        default = None,
        help='the dir to save log'
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
        '--freeze_layers',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--device',
        defalut='cpu',
        help='device id (i.e. 0 or 0,1 or cpu)'
    )

    args,unparsed = parser.parse_known_args()

    main(args)