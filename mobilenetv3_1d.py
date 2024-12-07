import timm
import torch
from torch import nn

sizes = [
    "mobilenetv3_large_075",
    "mobilenetv3_large_100",
    "mobilenetv3_rw",
    "mobilenetv3_small_050",
    "mobilenetv3_small_075",
    "mobilenetv3_small_100",
    "tf_mobilenetv3_large_075",
    "tf_mobilenetv3_large_100",
    "tf_mobilenetv3_large_minimal_100",
    "tf_mobilenetv3_small_075",
    "tf_mobilenetv3_small_100",
    "tf_mobilenetv3_small_minimal_100",
]


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.SiLU,
        gate_fn=torch.sigmoid,
        divisor=1,
        **_,
    ):
        super(SqueezeExcite, self).__init__()
        reduced_chs = reduced_base_chs
        self.conv_reduce = nn.Conv1d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv1d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2,), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class FastGlobalAvgPool1d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool1d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return (
                x.view(x.size(0), x.size(1), -
                       1).mean(-1).view(x.size(0), x.size(1), 1)
            )


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, drop, act, virtual_batch_size=32, momentum=0.1):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)
        self.drop = drop
        self.act = act

    def forward(self, x):
        # chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        # res = [self.bn(x_) for x_ in chunks]
        # return self.drop(self.act(torch.cat(res, dim=0)))
        # x = self.bn(x)
        # x = self.act(x)
        # x = self.drop(x)
        # return x
        return self.drop(self.act(self.bn(x)))


def replace_bn(parent):
    for n, m in parent.named_children():
        if type(m) is timm.layers.norm_act.BatchNormAct2d:
            # if type(m) is nn.BatchNorm2d:
            # print(type(m))
            setattr(
                parent,
                n,
                GBN(m.num_features, m.drop, m.act),
            )
        else:
            replace_bn(m)


def replace_se(parent):
    for n, m in parent.named_children():
        if type(m) is timm.models._efficientnet_blocks.SqueezeExcite:
            setattr(
                parent,
                n,
                SqueezeExcite(
                    m.conv_reduce.in_channels,
                    reduced_base_chs=m.conv_reduce.out_channels,
                ),
            )
        else:
            replace_se(m)


def replace_conv(parent, ds_rate):
    for n, m in parent.named_children():
        if type(m) is nn.Conv2d:
            if ds_rate == 2:
                setattr(
                    parent,
                    n,
                    nn.Conv1d(
                        m.in_channels,
                        m.out_channels,
                        kernel_size=m.kernel_size[0],
                        stride=m.stride[0],
                        padding=m.padding[0],
                        bias=m.kernel_size[0],
                        groups=m.groups,
                    ),
                )
            else:
                setattr(
                    parent,
                    n,
                    nn.Conv1d(
                        m.in_channels,
                        m.out_channels,
                        kernel_size=m.kernel_size[0] if m.kernel_size[0] == 1 else 5,
                        stride=m.stride[0] if m.stride[0] == 1 else ds_rate,
                        padding=m.padding[0] if m.padding[0] == 0 else 2,
                        bias=m.kernel_size[0],
                        groups=m.groups,
                    ),
                )
        else:
            replace_conv(m, ds_rate)


def create_mobilenetv3(network, ds_rate=2, in_chans=2):
    replace_se(network)
    replace_bn(network)
    replace_conv(network, ds_rate)
    network.global_pool = FastGlobalAvgPool1d()

    network.conv_stem = nn.Conv1d(
        in_channels=in_chans,
        out_channels=network.conv_stem.out_channels,
        kernel_size=network.conv_stem.kernel_size,
        stride=network.conv_stem.stride,
        padding=network.conv_stem.padding,
        bias=network.conv_stem.kernel_size,
        groups=network.conv_stem.groups,
    )

    return network


def mobilenetv3(
    model_size="mobilenetv3_small_050",
    num_classes: int = 10,
    drop_rate: float = 0,
    drop_path_rate: float = 0,
    in_chans=2,
):
    mdl = create_mobilenetv3(
        timm.create_model(
            model_size,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            exportable=True,
        ),
        in_chans=in_chans,
    )
    return mdl
