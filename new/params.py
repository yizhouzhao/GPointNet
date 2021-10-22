z_dim = 1024
point_num = 2048

e_l_steps  = 80 # number of langevin steps
e_l_step_size = 0.4 # stepsize of langevin
e_prior_sig = 1.0 # prior of ebm z
e_l_with_noise = True # noise term of langevin

batch_size = 16 # batch size 

g_l_steps = 40 # number of langevin steps
g_llhd_sigma = 0.3 # prior of factor analysis
g_l_step_size = 0.1 # stepsize of langevin
g_l_with_noise = True # noise term of langevin
