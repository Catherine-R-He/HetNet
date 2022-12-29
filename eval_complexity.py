from Net import Net
import dataset as dataset
import torch

cfg = dataset.Config(datapath='/home/crh/MirrorDataset/PMD', savepath='./out_resnext_PMD/v4/', mode='train', batch=12, lr=0.01, momen=0.9, decay=5e-4, epoch=300)


from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = Net(cfg)
  macs, params = get_model_complexity_info(net, (3, 352, 352), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))