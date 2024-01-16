import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Text, Dict
from pdb import set_trace as bp
from models import NetworkCIFAR as Network

__all__ = ['OPS', 'ResNetBasicblock', 'SearchSpaceNames']

class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
            )

    def forward(self, x):
        return self.op(x)


class DualSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(DualSepConv, self).__init__()
        self.op_a = SepConv(C_in, C_in , kernel_size, stride, padding, dilation, affine, track_running_stats)
        self.op_b = SepConv(C_in, C_out, kernel_size, 1, padding, dilation, affine, track_running_stats)

    def forward(self, x):
        x = self.op_a(x)
        x = self.op_b(x)
        return x


class ResNetBasicblock(nn.Module):

    def __init__(self, inplanes, planes, stride, affine=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine)
        self.conv_b = ReLUConvBN(  planes, planes, 3,      1, 1, 1, affine)
        if stride == 2:
            self.downsample = nn.Sequential(
                                                      nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                                      nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine)
        else:
            self.downsample = None
        self.in_dim  = inplanes
        self.out_dim = planes
        self.stride  = stride
        self.num_conv = 2

    def extra_repr(self):
        string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
        return string

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class POOLING(nn.Module):

    def __init__(self, C_in, C_out, stride, mode, affine=True, track_running_stats=True):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine, track_running_stats)
        if mode == 'avg'  : self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max': self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else              : raise ValueError('Invalid mode={:} in POOLING'.format(mode))

    def forward(self, inputs):
        if self.preprocess: x = self.preprocess(inputs)
        else              : x = inputs
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in   = C_in
        self.C_out  = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1: return x.mul(0.)
            else               : return x[:,:,::self.stride,::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in   = C_in
        self.C_out  = C_out
        self.relu   = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append( nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False) )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


OPS = {
    'none'        : lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
    'avg_pool_3x3': lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, 'avg', affine, track_running_stats),
    'max_pool_3x3': lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, 'max', affine, track_running_stats),
    'nor_conv_7x7': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (7,7), (stride,stride), (3,3), (1,1), affine, track_running_stats),
    'nor_conv_3x3': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1), affine, track_running_stats),
    'nor_conv_1x1': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (1,1), (stride,stride), (0,0), (1,1), affine, track_running_stats),
    'sep_conv_3x3': lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1), affine, track_running_stats),
    'sep_conv_5x5': lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(C_in, C_out, (5,5), (stride,stride), (2,2), (1,1), affine, track_running_stats),
    'dil_conv_3x3': lambda C_in, C_out, stride, affine, track_running_stats:     SepConv(C_in, C_out, (3,3), (stride,stride), (2,2), (2,2), affine, track_running_stats),
    'dil_conv_5x5': lambda C_in, C_out, stride, affine, track_running_stats:     SepConv(C_in, C_out, (5,5), (stride,stride), (4,4), (2,2), affine, track_running_stats),
    'skip_connect': lambda C_in, C_out, stride, affine, track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}


INF = 1000


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class NAS201SearchCell(nn.Module):

    def __init__(self, C_in, C_out, stride, max_nodes, op_names, affine=False, track_running_stats=True):
        super(NAS201SearchCell, self).__init__()

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                if j == 0:
                    xlists = [OPS[op_name](C_in, C_out, stride, affine, track_running_stats) for op_name in op_names]
                else:
                    xlists = [OPS[op_name](C_in, C_out, 1, affine, track_running_stats) for op_name in op_names]
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
        return string

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(sum(layer(nodes[j]) * w if w > 0.01 else 0 for layer, w in zip(self.edges[node_str], weights)))  # for pruning purpose
            nodes.append(sum(inter_nodes))
        return nodes[-1]


class MixedOp(nn.Module):

    def __init__(self, space, C, stride, affine, track_running_stats):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in space:
            op = OPS[primitive](C, C, stride, affine, track_running_stats)
            self._ops.append(op)

    def forward_darts(self, x, weights):
        return sum(w * op(x) if w > 0.01 else 0 for w, op in zip(weights, self._ops))  # for pruning purpose


# Learning Transferable Architectures for Scalable Image Recognition, CVPR 2018
class SearchCell(nn.Module):

    def __init__(self, space, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, affine, track_running_stats):
        super(SearchCell, self).__init__()
        self.reduction = reduction
        self.op_names = deepcopy(space)
        if reduction_prev: self.preprocess0 = OPS['skip_connect'](C_prev_prev, C, 2, affine, track_running_stats)
        else: self.preprocess0 = OPS['nor_conv_1x1'](C_prev_prev, C, 1, affine, track_running_stats)
        self.preprocess1 = OPS['nor_conv_1x1'](C_prev, C, 1, affine, track_running_stats)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self.edges = nn.ModuleDict()
        for i in range(self._steps):
            for j in range(2+i):
                node_str = '{:}<-{:}'.format(i, j)  # indicate the edge from node-(j) to node-(i+2)
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(space, C, stride, affine, track_running_stats)
                self.edges[ node_str ] = op
        self.edge_keys  = sorted(list(self.edges.keys()))
        self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
        self.num_edges  = len(self.edges)

    def forward_darts(self, s0, s1, weightss, alphass):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            clist = []
            for j, h in enumerate(states):
                node_str = '{:}<-{:}'.format(i, j)
                op = self.edges[ node_str ]
                weights = weightss[ self.edge2index[node_str] ]
                alphas = alphass[ self.edge2index[node_str] ]
                if sum(alphas) <= (-INF) * len(alphas):
                    # all ops on this edge are masked out
                    clist.append( 0 )
                else:
                    clist.append( op.forward_darts(h, weights) )
            states.append( sum(clist) )

        return torch.cat(states[-self._multiplier:], dim=1)

# The macro structure is based on NASNet
class NASNetworkDARTS(nn.Module):

    def __init__(self, C: int, N: int, steps: int, multiplier: int, stem_multiplier: int,
                 num_classes: int, search_space: List[Text], affine: bool, track_running_stats: bool,
                 depth=-1, use_stem=True):
        super(NASNetworkDARTS, self).__init__()
        self._C        = C
        self._layerN   = N  # number of stacked cell at each stage
        self._steps    = steps
        self._multiplier = multiplier
        self.depth = depth
        self.use_stem = use_stem
        self.stem = nn.Sequential(
           nn.Conv2d(3 if use_stem else min(3, C), C*stem_multiplier, kernel_size=3, padding=1, bias=False),
           nn.BatchNorm2d(C*stem_multiplier))

        # config for each layer
        layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * (N-1) + [C*4 ] + [C*4  ] * (N-1)
        layer_reductions = [False] * N + [True] + [False] * (N-1) + [True] + [False] * (N-1)

        num_edge, edge2index = None, None
        C_prev_prev, C_prev, C_curr, reduction_prev = C*stem_multiplier, C*stem_multiplier, C, False

        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if depth > 0 and index >= depth: break
            cell = SearchCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine, track_running_stats)
            if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
            else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append( cell )
            C_prev_prev, C_prev, reduction_prev = C_prev, multiplier*C_curr, reduction
        self.op_names   = deepcopy( search_space )
        self._Layer     = len(self.cells)
        self.edge2index = edge2index
        self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        num_concepts = 8
        self.concepts_layer = nn.Linear(C_prev, num_concepts)
        self.classifier = nn.Linear(num_concepts, num_classes)
        self.arch_normal_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)) )
        self.arch_reduce_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)) )

    def get_weights(self) -> List[torch.nn.Parameter]:
        xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
        xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
        xlist+= list( self.classifier.parameters() )
        return xlist

    def get_alphas(self) -> List[torch.nn.Parameter]:
        return [self.arch_normal_parameters, self.arch_reduce_parameters]

    def set_alphas(self, arch_parameters):
        self.arch_normal_parameters.data.copy_(arch_parameters[0].data)
        self.arch_reduce_parameters.data.copy_(arch_parameters[1].data)

    def show_alphas(self) -> Text:
        with torch.no_grad():
            A = 'arch-normal-parameters :\n{:}'.format( nn.functional.softmax(self.arch_normal_parameters, dim=-1).cpu() )
            B = 'arch-reduce-parameters :\n{:}'.format( nn.functional.softmax(self.arch_reduce_parameters, dim=-1).cpu() )
        return '{:}\n{:}'.format(A, B)

    def get_message(self) -> Text:
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self) -> Text:
        return ('{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

    def genotype2cmd(self, genotype):
        cmd = "Genotype(normal=%s, normal_concat=[2, 3, 4, 5], reduce=%s, reduce_concat=[2, 3, 4, 5])"%(genotype['normal'], genotype['reduce'])
        return cmd

    def genotype(self) -> Dict[Text, List]:
        def _parse(weights):
            gene = []
            n = 2; start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                selected_edges = []
                _edge_indice = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != self.op_names.index('none')))[:2]
                for _edge_index in _edge_indice:
                    _op_indice = list(range(W.shape[1]))
                    _op_indice.remove(self.op_names.index('none'))
                    _op_index = sorted(_op_indice, key=lambda x: -W[_edge_index][x])[0]
                    selected_edges.append( (self.op_names[_op_index], _edge_index) )
                gene += selected_edges
                start = end; n += 1
            return gene
        with torch.no_grad():
            gene_normal = _parse(torch.softmax(self.arch_normal_parameters, dim=-1).cpu().numpy())
            gene_reduce = _parse(torch.softmax(self.arch_reduce_parameters, dim=-1).cpu().numpy())
        return self.genotype2cmd({'normal': gene_normal, 'normal_concat': list(range(2+self._steps-self._multiplier, self._steps+2)), 'reduce': gene_reduce, 'reduce_concat': list(range(2+self._steps-self._multiplier, self._steps+2))})

    def forward(self, inputs):

        normal_w = nn.functional.softmax(self.arch_normal_parameters, dim=1)
        reduce_w = nn.functional.softmax(self.arch_reduce_parameters, dim=1)
        normal_a = self.arch_normal_parameters.detach().clone()
        reduce_a = self.arch_reduce_parameters.detach().clone()

        if self.use_stem:
            s0 = s1 = self.stem(inputs)
        else:
            s0 = s1 = inputs
        for i, cell in enumerate(self.cells):
            if cell.reduction: ww, aa = reduce_w, reduce_a
            else             : ww, aa = normal_w, normal_a
            s0, s1 = s1, cell.forward_darts(s0, s1, ww, aa)
        out = self.lastact(s1)
        out = self.global_pooling(out)
        #print(f"out step1: {out}")
        #print(f"out size1: {out.shape}")
        out = out.view(out.size(0), -1)
        #print(f"out step2: {out}")
        #print(f"out size2: {out.shape}")
        out8 = self.concepts_layer(out)
        #print(f"out8 step1: {out8}")
        #print(f"out8 size1: {out8.shape}")
        logits = self.classifier(out8)
        #print(f"logits step1: {logits}")
        #print(f"logits size1: {logits.shape}")

        return out, logits, out8


DARTS_SPACE   = ['none', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'avg_pool_3x3', 'max_pool_3x3']

#model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)

model = NASNetworkDARTS(C=3, N=1, steps=4, multiplier=4, num_classes=2, search_space=DARTS_SPACE, affine=True,stem_multiplier=1, track_running_stats=False)
#model = torch.load('model_results/model.pth')
model_state_dict = torch.load('model_state_dict_results/model.pth')

#print(model_state_dict['cells.0.preprocess1.op.2.weight'])

model.load_state_dict(model_state_dict)

#print(model)

#print(f"cells.2.preprocess1.op.2.weight")    