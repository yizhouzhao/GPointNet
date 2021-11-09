###################   network    ############################
z_dim = 128 # may change to 1024
point_num = 2048

e_l_steps  = 30 # number of langevin steps
e_l_step_size = 0.05 # stepsize of langevin
e_prior_sig = 1.0 # prior of ebm z
e_l_with_noise = True # noise term of langevin
e_energy_form = "identity"
e_decay = 0
e_beta1 = 0.05
e_beta2 = 0.999

g_l_steps = 10 # number of langevin steps
g_llhd_sigma = 0.3 # prior of factor analysis
g_l_step_size = 0.05 # stepsize of langevin
g_l_with_noise = True # noise term of langevin
g_decay = 0
g_beta1 = 0.5
g_beta2 = 0.999

###################   Training    ############################
batch_size = 16 # batch size 
n_epochs = 400 # epochs
gpu_deterministic = False #set cudnn in deterministic mode (slow)

e_lr = 2e-5 # ebm learning rate
g_lr = 1e-4 # gen learning rate

# e_gamma = 0.998 # lr decay for ebm
# g_gamma = 0.998 # lr decay for gen

e_init_sig = 1 # sigma of initial distribution
g_init_sig = 1

################# Log ###################
n_printout = 25 # print every

################# Sythesis ###################
langevin_clip = 1.0

################# Channel ####################
e_alpha = 1.0
e_beta = 0.1
e_gamma = 0.01

g_alpha = 1.0
g_beta = 0.1
g_gamma = 0.01
g_delta = 1.0


