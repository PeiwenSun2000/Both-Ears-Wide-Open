import argparse
import numpy as np

def init_args(return_parser=False): 
    parser = argparse.ArgumentParser(description="""Configure""")

    # basic configuration 
    parser.add_argument('--exp', type=str, default='test101',
                        help='checkpoint folder')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of total epochs to run (default: 90)')

    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--resume_optim', default=False, action='store_true')
    parser.add_argument('--save_step', default=1, type=int)
    parser.add_argument('--valid_step', default=1, type=int)
    

    # Dataloader parameter
    parser.add_argument('--max_sample', default=-1, type=int)
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', default=24, type=int)

    # network parameters
    parser.add_argument('--setting', type=str, default='', required=False)

    parser.add_argument('--backbone', type=str, default='resnet1d', required=False)
    # parser.add_argument('--pretrained', default=False, action='store_true')

    # parser.add_argument('--no_bn', default=False, action='store_true')
    # optimizer parameters
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', default=1e-4,
                        type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--optim', type=str, default='Adam',
                        choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--schedule', type=str, default='cos', choices=['none', 'cos', 'step'], required=False)
    # Loss parameters
    parser.add_argument('--loss_type', type=str, default='MSE', choices=['BCE', 'CE', 'MSE', 'MAE', 'Huber'], required=False)
    parser.add_argument('--no_bn', default=False, action='store_true')
    # parser.add_argument('--aug_spec', default=False, action='store_true')
    parser.add_argument('--aug_wave', default=False, action='store_true')
    parser.add_argument('--regular_scaling', default=False, action='store_true')

    parser.add_argument('--shift_wave', default=False, action='store_true')
    parser.add_argument('--larger_shift', default=False, action='store_true')
    
    parser.add_argument('--aug_img', default=False, action='store_true')

    parser.add_argument('--normalized_rms', default=False, action='store_true')
    parser.add_argument('--valid_by_step', default=False, action='store_true')

    parser.add_argument('--test_mode', default=False, action='store_true')
    parser.add_argument('--no_resample', default=False, action='store_true')
    parser.add_argument('--add_color_jitter', default=False, action='store_true')


    parser.add_argument('--patch_size', type=int, default=None, required=False)
    parser.add_argument('--patch_stride', type=int, default=None, required=False)
    parser.add_argument('--patch_num', type=int, default=None, required=False)
    parser.add_argument('--skip_node', default=False, action='store_true')
    parser.add_argument('--fake_right', default=False, action='store_true')
    # parser.add_argument('--add_noise', default=False, action='store_true')

    parser.add_argument('--wav2spec', default=False, action='store_true')
    parser.add_argument('--tau', type=float, default=0.05, required=False)

    parser.add_argument('--cycle_num', type=int, default=1, required=False)
    parser.add_argument('--crw_rate', type=float, default=1.0, required=False)
    parser.add_argument('--synthetic_rate', type=float, default=0.0, required=False)
    parser.add_argument('--teacher_rate', type=float, default=0.0, required=False)
    parser.add_argument('--clip_length', type=float, default=0.96, required=False)

    parser.add_argument('--large_feature_map', default=False, action='store_true')
    parser.add_argument('--add_sounds', type=int, default=0, required=False)
    parser.add_argument('--max_weight', type=float, default=0, required=False)

    parser.add_argument('--smooth', type=float, default=0.0, required=False)
    parser.add_argument('--noiseSNR', type=float, default=None, required=False)
    parser.add_argument('--bidirectional', default=False, action='store_true')

    parser.add_argument('--no_baseline', default=False, action='store_true')
    parser.add_argument('--baseline_type', type=str, default='mean', choices=['gcc_phat', 'advgcc', 'flipcoin', 'visualvoice'], required=False)

    parser.add_argument('--mode', type=str, default='mean', choices=['mean', 'ransac'], required=False)
    parser.add_argument('--select', type=str, default='soft_weight', choices=['soft_weight', 'argmax'], required=False)
    parser.add_argument('--gcc_fft', type=int, default=1024, required=False)

    parser.add_argument('--cycle_filter', default=False, action='store_true')
    parser.add_argument('--img_feat_scaling', type=float, default=None, required=False)
    parser.add_argument('--max_delay', type=float, default=None, required=False)
    parser.add_argument('--correspondence_type', type=str, default='mean', choices=['mean', 'max'], required=False)

    parser.add_argument('--add_reverb', default=False, action='store_true')
    parser.add_argument('--add_mixture', default=False, action='store_true')
    parser.add_argument('--ignore_speaker', default=False, action='store_true')
    parser.add_argument('--crop_face', default=False, action='store_true')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to checkpoint (default: None)')

    parser.add_argument('--same_vote', default=False, action='store_true')
    parser.add_argument('--same_noise', default=False, action='store_true')
    parser.add_argument('--finer_hop', default=False, action='store_true')




    if return_parser:
        return parser

    # global args
    args = parser.parse_args()

    return args
