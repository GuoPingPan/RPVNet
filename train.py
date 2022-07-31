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


    print('-----------------')
    print(f'dataset dir: {args.dataset}')
    print(f'log dir: {args.log}')
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

    # create log dir for saving model
    if args.log is None:
        root,_ = os.path.split(os.path.abspath(__file__))
        default= root + '/log/' + \
                 datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/'
        print(default)
        print(f'The log dir [{args.log}] doesn\'t exist, Do you want to use'
              f'[{default}] as default? [y/n]')
        sig = input()
        if sig in ['Y','y']:
            args.log = default
            if not os.path.exists(root+'/log/'): os.mkdir(root+'/log/')
            if not os.path.exists(default): os.mkdir(default)
            else:
                print(f'Check the log dir.')
                quit()
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
                    os.mkdir(args.log)
                else:
                    print(f'Check the log dir.')
                    quit()
        else:
            print(f'Using the dir [{args.log}] to contains the log? [y/n]')
            sig = input()
            if sig in ['Y', 'y']:
                os.mkdir(args.log)
            else:
                print(f'Check the log dir.')
                quit()

    # check if use pretrained model
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint) and args.checkpoint.endswith('.ckpt'):
            print(f'Using the pretrained model:[{args.checkpoint}]')


    trainer = Trainer(args=args,model_cfg=model_cfg,data_cfg=data_cfg)
    trainer.train()




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
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='device id (i.e. 0 or 0,1 or cuda)'
    )

    args,unparsed = parser.parse_known_args()

    main(args)