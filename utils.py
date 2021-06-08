import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

__all__ = ['check_sparsity']

def check_sparsity(model, conv1=True):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    sum_list = sum_list + float(m.weight.nelement())
                    zero_sum = zero_sum + float(torch.sum(m.weight == 0))
                else:
                    print('skip conv1 for sparsity checking')
            else:
                sum_list = sum_list + float(m.weight.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    print('* remain weight = ', 100 * (1 - zero_sum / sum_list), '%')

    return 100 * (1 - zero_sum / sum_list)

def speed_up_inference_for_channel(mm:nn.Module):
    # assert pruned dim is dim=0
    # TODO: get better performance for shortcut
    # TODO: WARNING!!!!! after use this function,
    #  the backward method are not well defined, make sure torch.no_grad() is called
    if not (isinstance(mm,nn.Conv2d) or isinstance(mm,nn.Linear)):
        # only conv2d or linear are pruned
        return

    if not hasattr(mm,'weight_mask'):
        # mm is unpruned or accelerated
        return

    if hasattr(mm,'bias') and isinstance(mm.bias, nn.Parameter):
        #print(mm.bias)
        return
        #return

    def set_weight(md:nn.Module,input):
        #print('set_weight')
        if not hasattr(md,'output_len'):
            #print('set_len')
            output_len = md.weight.size(0)
            md.register_buffer('output_len',torch.ones(1)*output_len)
            mask_list = md.weight_mask[:, 0, 0, 0]
            pl = torch.where(mask_list > 0)[0]
            tmp_weight = md.weight[pl]
            for k, hook in md._forward_pre_hooks.items():
                if isinstance(hook, prune.BasePruningMethod) and hook._tensor_name == "weight":
                    hook.remove(md)
                    del md._forward_pre_hooks[k]
                    break
            md.weight = nn.Parameter(tmp_weight)
            md.register_buffer('scatter_place',pl)


    def set_scatter(md:nn.Module,input,output):
        sizes = list(output.shape)
        #print(sizes)
        sizes[1] = int(md.output_len.item())
        out = output.new_zeros(sizes)
        out[:,md.scatter_place] = output
        return out

    mm.register_forward_pre_hook(set_weight)
    mm.register_forward_hook(set_scatter)




