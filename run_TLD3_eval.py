import rospy
import sys
sys.path.append('../../')
from rand_eval_gpu import RandEvalGpu
from utility import *


def evaluate_TLD3(pos_start=0, pos_end=199, model_name='TLD3', save_dir='../saved_model/',
                  state_num=22, is_scale=True, is_poisson=False, is_save_result=True,
                  use_cuda=True):

    rospy.init_node('TLD3_eval')
    poly_list, raw_poly_list = gen_test_env_poly_list_env()
    start_goal_pos = pickle.load(open(".../eval_positions.p", "rb"))
    robot_init_list = start_goal_pos[0][pos_start:pos_end + 1]
    goal_list = start_goal_pos[1][pos_start:pos_end + 1]
    actor_net = load_test_actor_network( state_num=state_num)
    eval = RandEvalGpu(actor_net, robot_init_list, goal_list, poly_list,
                       max_steps=1000, action_rand=0.01, goal_dis_min_dis=0.3,
                       is_scale=is_scale, is_poisson=is_poisson, use_cuda=use_cuda)
    data = eval.run_ros()
    if is_save_result:
        pickle.dump(data,
                    open('...record_data/' + model_name + '_' + str(pos_start) + '_' + str(pos_end) + '.p', 'wb+'))
    print(str(model_name) + " Eval on GPU Finished ...")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--poisson', type=int, default=1)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    IS_POISSON = False
    STATE_NUM = 22
    MODEL_NAME = 'TLD3'
    if args.poisson == 1:
        IS_POISSON = True
        STATE_NUM = 24
        MODEL_NAME = 'TLD3_poisson'
    SAVE_RESULT = True
    if args.save == 1:
        SAVE_RESULT = True
    evaluate_TLD3(use_cuda=USE_CUDA, state_num=STATE_NUM,
                  is_poisson=IS_POISSON, model_name=MODEL_NAME,
                  is_save_result=SAVE_RESULT)
