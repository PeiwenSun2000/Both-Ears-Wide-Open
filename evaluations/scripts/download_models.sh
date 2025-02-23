#!/bin/bash
{
mkdir -p checkpoints/pretrained-models
prefix='checkpoints/pretrained-models'

wget https://www.dropbox.com/s/tgg7fkusih1jp0q/FreeMusic-MonoCLR-1024.pth.tar?dl=0 -O $prefix/FreeMusic-MonoCLR-1024.pth.tar
wget https://www.dropbox.com/s/zcum0zfssnfstdi/FreeMusic-ZeroNCE-1024.pth.tar?dl=0 -O $prefix/FreeMusic-ZeroNCE-1024.pth.tar
wget https://www.dropbox.com/s/qwepkmli4cifn84/FreeMusic-StereoCRW-1024.pth.tar?dl=0 -O $prefix/FreeMusic-StereoCRW-1024.pth.tar

wget https://www.dropbox.com/s/72bvqo6nspklgih/VoxCeleb2-AVMonoCLR-15360.pth.tar?dl=0 -O $prefix/VoxCeleb2-AVMonoCLR-15360.pth.tar

wget https://www.dropbox.com/s/l0rcktgl9s8dlqy/FreeMusic-MonoCLR-7680.pth.tar?dl=0 -O $prefix/FreeMusic-MonoCLR-7680-training-use.pth.tar
wget https://www.dropbox.com/s/hwocipd1c8mk5y0/FreeMusic-StereoCRW-7680.pth.tar?dl=0 -O $prefix/FreeMusic-StereoCRW-7680-training-use.pth.tar
wget https://www.dropbox.com/s/6ibxc6lhxx68vof/FreeMusic-ZeroNCE-7680.pth.tar?dl=0 -O $prefix/FreeMusic-ZeroNCE-7680-training-use.pth.tar

}

