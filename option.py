import argparse

parser = argparse.ArgumentParser(description='CMA_XD_VioDet')
parser.add_argument('--rgb-list', default='list/rgb.list', help='list of rgb features ')
parser.add_argument('--flow-list', default='list/flow.list', help='list of flow features')
parser.add_argument('--audio-list', default='list/audio.list', help='list of audio features')
parser.add_argument('--test-rgb-list', default='list/rgb_test.list', help='list of test rgb features ')
parser.add_argument('--test-flow-list', default='list/flow_test.list', help='list of test flow features')
parser.add_argument('--test-audio-list', default='list/audio_test.list', help='list of test audio features')
parser.add_argument('--dataset-name', default='XD-Violence', help='dataset to train on XD-Violence')
parser.add_argument('--gt', default='list/gt.npy', help='file of ground truth ')


parser.add_argument('--modality', default='MIX2', help='the type of the input, AUDIO,RGB,FLOW, MIX1, MIX2, '
                                                          'or MIX3, MIX_ALL')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0005)')
parser.add_argument('--batch-size', type=int, default=128, help='number of instances in a batch of data')
parser.add_argument('--workers', default=8, help='number of workers in dataloader')
parser.add_argument('--model-name', default='xd_a2v', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', type=int, default=1024+128, help='size of feature (default: 2048)')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--max-seqlen', type=int, default=200, help='maximum sequence length during training')
parser.add_argument('--max-epoch', type=int, default=50, help='maximum iteration to train (default: 50)')
parser.add_argument('--seed', type=int, default=9, help='Random Initiation (default: 9)')
