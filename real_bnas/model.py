import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES, PRIMITIVES_reduce
from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride, reduction):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.reduction = reduction

        if self.reduction == True:
            PRIMITIVES_MixedOp = PRIMITIVES_reduce
        else:
            PRIMITIVES_MixedOp = PRIMITIVES
        for primitive in PRIMITIVES_MixedOp:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, layer_no, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.layer = layer_no
        self.reduction = reduction
        self.reduction_prev = reduction_prev

        if reduction_prev:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 3, 2, 1, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        if self.reduction:
            self.preprocess_res = nn.Sequential(nn.BatchNorm2d(C_prev, affine=True),
                                                nn.Conv2d(C_prev, 4 * C, kernel_size=2, stride=2))
            self.preprocess_res_xpad = nn.Sequential(nn.BatchNorm2d(C_prev, affine=True),
                                                nn.Conv2d(C_prev, 4 * C, kernel_size=2, stride=2, padding=(1, 0)))
            self.preprocess_res_ypad = nn.Sequential(nn.BatchNorm2d(C_prev, affine=True),
                                    nn.Conv2d(C_prev, 4 * C, kernel_size=2, stride=2, padding=(0, 1)))

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.reduction)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        res0 = s0
        res1 = s1
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        if self.layer == 0:
            states_out = torch.cat(states[-self._multiplier:], dim=1)
            return states_out
        else:
            if self.reduction:
                states_out = torch.cat(states[-self._multiplier:], dim=1)
                if res1.shape[2] % 2 == 1 and res1.shape[3] % 2 == 0:
                    preprocess_out = self.preprocess_res_xpad(res1)
                elif res1.shape[2] % 2 == 0 and res1.shape[3] % 2 == 1:
                    preprocess_out = self.preprocess_res_ypad(res1)
                else:
                    preprocess_out = self.preprocess_res(res1)
                states_out += preprocess_out
                return states_out
            else:

                states_out = torch.cat(states[-self._multiplier:], dim=1)
                states_out += res1
                return states_out

class Network(nn.Module):

    def __init__(self, C, data_channel, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        C_curr = stem_multiplier * C

        self.stem = ReLUConvBN(data_channel, C_curr, 3, stride=1, padding=1)

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, i, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops_normal = len(PRIMITIVES)
        num_ops_reduce = len(PRIMITIVES_reduce)
        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops_normal).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops_reduce).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self, gamma=1.):

        def _parse(weights, PRIMITIVES_parse):

            gene = []
            n = 2
            start = 0
            weights[0, :] /= gamma
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES_parse[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), PRIMITIVES_parse=PRIMITIVES)
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),
                             PRIMITIVES_parse=PRIMITIVES_reduce)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype