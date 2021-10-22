import argparse

def parse_config():

    parser = argparse.ArgumentParser()

    # Point cloud related
    parser.add_argument('-net_type', type=str, default="default_medium")
    parser.add_argument('-num_point', type=int, default=2048)
    parser.add_argument('-point_dim', type=int, default=3)
    parser.add_argument('-argment_mode', type=int, default=0)
    parser.add_argument('-argment_noise', type=float, default=0.01)
    parser.add_argument('-random_sample', type=int, default=1)
    parser.add_argument('-visualize_mode', type=int, default=0)
    parser.add_argument('-learning_mode', type=int, default=0)
    parser.add_argument('-normalize', type=str, default="ebp")
    parser.add_argument('-batch_norm', type=str, default="ln", help='BatchNorm(bn) / LayerNorm(ln) / InstanceNorm(in) / None')
    parser.add_argument('-activate_eval', type=int, default=0)
     
    # EBM related
    parser.add_argument('-batch_size', type=int, default=128, help='')
    parser.add_argument('-lr', type=float, default=0.0005, help='')
    parser.add_argument('-lr_decay', type=float, default=0.998, help='')
    parser.add_argument('-beta1_des', type=float, default=0.9, help='')
    parser.add_argument('-sample_step', type=int, default=64, help='')
    parser.add_argument('-activation', type=str, default="ReLU", help='')
    parser.add_argument('-step_size', type=float, default=0.01, help='')
    parser.add_argument('-noise_decay', type=int, default=0, help='')
    parser.add_argument('-langevin_decay', type=int, default=0, help='')
    parser.add_argument('-ref_sigma', type=float, default=0.3, help='')
    parser.add_argument('-num_chain', type=float, default=1, help='')
    parser.add_argument('-langevin_clip', type=float, default=1, help='')
    parser.add_argument('-warm_start', type=int, default=0)

    # Logistic related
    parser.add_argument('-num_steps', type=int, default=2000)
    parser.add_argument('-stable_check', type=int, default=1)
    parser.add_argument('-do_evaluation', type=int, default=1)
    parser.add_argument('-seed', type=int, default=666, help='')
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('-eval_step', type=int, default=50)
    parser.add_argument('-drop_last', action='store_true')
    parser.add_argument('-mode', type=str, default="train", help='')
    parser.add_argument('-data_size', type=int, default=10000)
    parser.add_argument('-test_size', type=int, default=16)
    parser.add_argument('-debug', type=int, default=99, help='')
    parser.add_argument('-cuda', type=str, default="-1", help='')
    parser.add_argument('-data_path', type=str, default="data")
    parser.add_argument('-checkpoint_path', type=str, default="")
    parser.add_argument('-category', type=str, default="chair")
    parser.add_argument('-output_dir', type=str, default="default")
    parser.add_argument('-fp16', type=str, default="None", help='/O1/O2')
    
    return parser.parse_args("")
