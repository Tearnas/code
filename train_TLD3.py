import rospy
import time
import os
from torch.utils.tensorboard import SummaryWriter
import sys
import pickle

#sys.path.append('../../')
from TLD3_agent import Agent
from environment import GazeboEnvironment
from utility import *


def train_TLD3(run_name="TLD3_R1", exp_name="Rand_R1", episode_num=(100, 200, 300, 400),
               iteration_num_start=(200, 300, 400, 500), iteration_num_step=(1, 2, 3, 4),
               iteration_num_max=(1000, 1000, 1000, 1000),
               linear_spd_max=0.5, linear_spd_min=0.05, save_steps=10000,
               env_epsilon=(0.9, 0.6, 0.6, 0.6), env_epsilon_decay=(0.999, 0.9999, 0.9999, 0.9999),
               laser_half_num=9, laser_min_dis=0.35, scan_overall_num=36, goal_dis_min_dis=0.3,
               obs_reward=-20, goal_reward=30, goal_dis_amp=15, goal_th=0.5, obs_th=0.35,
               state_num=22, action_num=2, is_pos_neg=False, is_poisson=False, poisson_win=50,
               memory_size=100000, batch_size=256, epsilon_end=0.1, rand_start=10000, rand_decay=0.999,
               rand_step=2, target_tau=0.01, target_step=1, use_cuda=True):

    dirName = 'save_TLD3_weights'
    try:
        os.mkdir('../' + dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")


    env1_poly_list, env1_raw_poly_list, env1_goal_list, env1_init_list = gen_rand_list_env1(episode_num[0])
    env2_poly_list, env2_raw_poly_list, env2_goal_list, env2_init_list = gen_rand_list_env2(episode_num[1])
    env3_poly_list, env3_raw_poly_list, env3_goal_list, env3_init_list = gen_rand_list_env3(episode_num[2])
    env4_poly_list, env4_raw_poly_list, env4_goal_list, env4_init_list = gen_rand_list_env4(episode_num[3])
    overall_poly_list = [env1_poly_list, env2_poly_list, env3_poly_list, env4_poly_list]

    overall_list = pickle.load(open(".../random_positions/" + exp_name + ".p", "rb"))
    overall_init_list = overall_list[0]
    overall_goal_list = overall_list[1]

    print("Use Training Rand Start and Goal Positions: ", exp_name)


    rospy.init_node("train_TLD3")
    env = GazeboEnvironment(laser_scan_half_num=laser_half_num, laser_scan_min_dis=laser_min_dis,
                            scan_dir_num=scan_overall_num, goal_dis_min_dis=goal_dis_min_dis,
                            obs_reward=obs_reward, goal_reward=goal_reward, goal_dis_amp=goal_dis_amp,
                            goal_near_th=goal_th, obs_near_th=obs_th)
    if is_pos_neg:
        rescale_state_num = state_num + 2
    else:
        rescale_state_num = state_num
    agent = Agent(state_num, action_num, rescale_state_num, poisson_window=poisson_win, use_poisson=is_poisson,
                  memory_size=memory_size, batch_size=batch_size, epsilon_end=epsilon_end,
                  epsilon_rand_decay_start=rand_start, epsilon_decay=rand_decay, epsilon_rand_decay_step=rand_step,
                  target_tau=target_tau, target_update_steps=target_step, use_cuda=use_cuda)

    tb_writer = SummaryWriter()


    overall_steps = 0
    overall_episode = 0
    env_episode = 0
    env_ita = 0
    ita_per_episode = iteration_num_start[env_ita]
    env.set_new_environment(overall_init_list[env_ita],
                            overall_goal_list[env_ita],
                            overall_poly_list[env_ita])
    agent.reset_epsilon(env_epsilon[env_ita],
                        env_epsilon_decay[env_ita])

    start_time = time.time()
    while True:
        state = env.reset(env_episode)
        rescale_state = ddpg_state_rescale(state, rescale_state_num)
        episode_reward = 0
        for ita in range(ita_per_episode):
            ita_time_start = time.time()
            overall_steps += 1
            raw_action = agent.act(rescale_state)
            decode_action = wheeled_network_2_robot_action_decoder(
                raw_action, linear_spd_max, linear_spd_min
            )
            next_state, reward, done = env.step(decode_action)
            rescale_next_state = ddpg_state_rescale(state, rescale_state_num)

            episode_reward += reward
            agent.remember(state, rescale_state, raw_action, reward, next_state, rescale_next_state, done)
            state = next_state
            rescale_state = rescale_next_state


            if len(agent.memory) > batch_size:
                actor_loss_value, critic_loss_value = agent.replay()
                tb_writer.add_scalar('TLD3/actor_loss', actor_loss_value, overall_steps)
                tb_writer.add_scalar('TLD3/critic_loss', critic_loss_value, overall_steps)
            ita_time_end = time.time()
            tb_writer.add_scalar('TLD3/ita_time', ita_time_end - ita_time_start, overall_steps)
            tb_writer.add_scalar('TLD3/action_epsilon', agent.epsilon, overall_steps)

            if overall_steps % save_steps == 0:
                agent.save("../save_ddpg_weights", overall_steps // save_steps, run_name)


            if done or ita == ita_per_episode - 1:
                print("Episode: {}/{}, Avg Reward: {}, Steps: {}"
                      .format(overall_episode, episode_num, episode_reward / (ita + 1), ita + 1))
                tb_writer.add_scalar('DDPG/avg_reward', episode_reward / (ita + 1), overall_steps)
                break
        if ita_per_episode < iteration_num_max[env_ita]:
            ita_per_episode += iteration_num_step[env_ita]
        if overall_episode == 999:
            agent.save("../save_ddpg_weights", 0, run_name)
        overall_episode += 1
        env_episode += 1
        if env_episode == episode_num[env_ita]:
            print("Environment ", env_ita, " Training Finished ...")
            if env_ita == 3:
                break
            env_ita += 1
            env.set_new_environment(overall_init_list[env_ita],
                                    overall_goal_list[env_ita],
                                    overall_poly_list[env_ita])
            agent.reset_epsilon(env_epsilon[env_ita],
                                env_epsilon_decay[env_ita])
            ita_per_episode = iteration_num_start[env_ita]
            env_episode = 0
    end_time = time.time()
    print("Finish Training with time: ", (end_time - start_time) / 60, " Min")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--poisson', type=int, default=1)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    IS_POS_NEG, IS_POISSON = False, False
    if args.poisson == 1:
        IS_POS_NEG, IS_POISSON = True, True
    train_TLD3(use_cuda=USE_CUDA, is_pos_neg=IS_POS_NEG, is_poisson=IS_POISSON)
