
import random
import json
import time
# TODO: 1. check times so change the hparams
# TODO: 2. check device is available
# TODO: 3. check method change
# TODO: 4. check hparam & hparam_list change change when method change
# TODO: 5. check exp_name change
# TODO: 6. do exp at different screen so you can early stop it

print('check times so change the hparams')
print('check device is available')
print('check method change')
print('check hparam & hparam_list change change when method change')
print('check exp_name change')

test_env = 1
method = 'ERM'
exp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
exp_name = 'ERM_prune_models'

steps = 5001
checkpoint_freq = 500
assert (steps-1)//10 == checkpoint_freq

resnet18 = False

hparams = {}

#mixup_list = [0.1,0.3]
#eta_list = [0.1,0.05,0.02]
hparams['lr'] = 5e-5
#hparams['mmd_gamma'] = 0.01
hparams['resnet18'] = resnet18
#hparams['cut_step'] = 50
#hparams['cut_percent'] = 0.02
#hparams['regular_lam'] = 1
#hparams['mixup_alpha'] = 0.2
#hparams['groupdro_eta'] = 0.01
#hparams['irm_lambda'] = 10
#hparams['irm_penalty_anneal_iters'] = 500
#hparams["vrex_penalty_anneal_iters"] = 400
#hparams["vrex_lambda"] = 1

#hparams['rsc_f_drop_factor'] = 0.2
#hparams['rsc_b_drop_factor'] = 0.2

# please make sure those devices are available
device = 0
str0 = 'mkdir logs/%s'%exp_name
print(str0)
for times in range(1):
    str0 = "CUDA_VISIBLE_DEVICES={} ".format(device)
    str0 += "python -m main " \
            "--data_dir domainbed/datasets \\\n"
    str0 += "--trial_seed {} ".format(random.randint(10000, 99999))

    str0 += "--algorithm {} --dataset OfficeHome --test_envs {} \\\n" \
            "--steps {} --output_dir logs/{}/ \\\n".format(method, test_env,steps, exp_name)
    arch = 'res18' if resnet18 else 'res50'
    raw_hparams = json.dumps(hparams)
    hparam_str = raw_hparams.replace('.', '*').replace(':', '=').replace(' ', '').replace(',', '_')
    file_name = "%s_%s_%s_%s " % (method, hparam_str, exp_time, times)
    str0 += "--checkpoint_freq {} \\\n".format(checkpoint_freq)
    #str0 += '--skip_model_save \\\n'
    #print(raw_hparams)
    str0 += '--pruning_method IMP \\\n'
    str0 += '--init_model init_model.pkl \\\n'
    raw_hparams = raw_hparams.replace('\"', '\\\"')
    str0 += "--hparams \"{}\" \\\n".format(raw_hparams)

    # str0 += "&"
    print(str0)
    print('\n')




