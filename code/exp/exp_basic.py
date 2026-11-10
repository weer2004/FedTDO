import os
import torch
from models import Dynamic_TMoE

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Dynamic_TMoE': Dynamic_TMoE
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        # GPU is mandatory project-wide: no CPU / mps fallback.  A row produced on
        # CPU is not comparable with the CUDA rows of the same table.
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required: CPU execution is disabled '
                               'project-wide.')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
        gpu_index = min(int(self.args.gpu or 0), torch.cuda.device_count() - 1)
        self.args.use_gpu = True
        self.args.gpu = gpu_index
        device = torch.device('cuda:{}'.format(gpu_index))
        print('Use GPU: cuda:{}'.format(gpu_index))
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
