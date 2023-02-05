import torch 
from typing import Tuple,List
import math
from torch.nn.utils.rnn import pad_sequence

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def pad_list(xs: List[torch.Tensor], pad_value: int):
    n_batch = len(xs)
    max_len = max([x.size(0) for x in xs])
    pad = torch.zeros(n_batch, max_len, dtype=xs[0].dtype, device=xs[0].device)
    pad = pad.fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def reverse_pad_list(ys_pad: torch.Tensor,
                     ys_lens: torch.Tensor,
                     pad_value: float = -1.0) -> torch.Tensor:
  

    r_ys_pad = pad_sequence([(torch.flip(y.int()[:i], [0]))
                             for y, i in zip(ys_pad, ys_lens)], True,
                            pad_value)
    return r_ys_pad

def add_sos_eos(ys_pad: torch.Tensor, sos: int, eos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)

def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp
    
