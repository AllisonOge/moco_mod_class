"""
Wrap model with MoCo.
"""
import torch
import torch.nn as nn
from mobilenetv3_1d import mobilenetv3


class MoCo_MobileNet(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, name='mobilenetv3_small', n_chans=2, dim=256, mlp_dim=1024, T=1.0):
        """
        :param name: the name of the base encoder
        :param n_chans: the number of input channels (default: 2)
        :param dim: the feature dimension (default: 256)
        :param mlp_dim: the hidden dimension of the MLP (default: 1024)
        :param T: the temperature (default: 1.0)
        """
        super(MoCo_MobileNet, self).__init__()
        self.T = T

        # build encoders
        self.base_encoder = mobilenetv3(
            name, in_chans=n_chans, num_classes=mlp_dim,
            drop_rate=0.2, drop_path_rate=0.7)
        self.momentum_encoder = mobilenetv3(
            name, in_chans=n_chans, num_classes=mlp_dim,
            drop_rate=0.2, drop_path_rate=0.7)
        hidden_dim = self.base_encoder.classifier.weight.shape[1]

        del self.base_encoder.classifier, self.momentum_encoder.classifier  # remove the classifier layer # noqa

        # projectors
        self.base_encoder.classifier = self._build_mlp(
            2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.classifier = self._build_mlp(
            2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k, dst=False):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        if dst:
            # gather all targets
            k = concat_all_gather(k)
            # Einstein sum is more intuitive
            logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
            N = logits.shape[0]  # batch size per GPU
            labels = (torch.arange(N, dtype=torch.long) +
                      N * torch.distributed.get_rank()).cuda()
        else:
            logits = torch.einsum("nc,mc->nm", [q, k]) / self.T
            labels = torch.arange(
                logits.shape[0], dtype=torch.long).to(logits.device)
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m, dst=False):
        """
        Input:
            x1: first view
            x2: second view
            m: moco momentum
        Output:
            loss
        """
        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2, dst) + self.contrastive_loss(q2, k1, dst)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
