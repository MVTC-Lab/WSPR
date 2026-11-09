import os
import torch

# Release-only registry: the six primary model classes.
from models import (PatchTST, iTransformer, WSPR, DirectNWP,
                    DirectNWP_iTransformer, WSPR_iTransformer)


class Exp_Basic(object):

    def __init__(self, args):
        self.args = args

        self.model_dict = {
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'WSPR': WSPR,
            'DirectNWP': DirectNWP,
            'DirectNWP_iTransformer': DirectNWP_iTransformer,
            'WSPR_iTransformer': WSPR_iTransformer,
        }
        if args.model not in self.model_dict:
            raise ValueError("This release supports only the six primary model classes.")

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)



    def _build_model(self):
        raise NotImplementedError
        return None


    def _acquire_device(self):

        if self.args.use_gpu and self.args.gpu_type == 'cuda':

            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu
            ) if not self.args.use_multi_gpu else self.args.devices


            device = torch.device(
                'cuda:{}'.format(self.args.gpu)
            )

            print('Use GPU: cuda:{}'.format(self.args.gpu))


        elif self.args.use_gpu and self.args.gpu_type == 'mps':

            device = torch.device('mps')

            print('Use GPU: mps')


        else:

            device = torch.device('cpu')

            print('Use CPU')


        return device


    def _get_data(self):
        pass


    def vali(self):
        pass


    def train(self):
        pass


    def test(self):
        pass
