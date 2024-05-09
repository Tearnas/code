import math
import random
import numpy as np
from shapely.geometry import Point, Polygon


def gen_rand_list_env1(size, robot_goal_diff=1.0):

    robot_init_pose = [8, 8]
    env_range = ((4, 12), (4, 12))
    poly_raw_list = [[(2, 14), (14, 14), (14, 2), (2, 2)]]
    goal_raw_list = [[(4, 12), (12, 12), (12, 4), (4, 4)]]

    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)

    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            goal = random.choice(goal_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list


def gen_rand_list_env2(size, robot_goal_diff=3.0):

    robot_init_pose_range = ((5.75, 10.25), (-11, -5))
    robot_init_pose_list = []
    init_x, init_y = np.mgrid[robot_init_pose_range[0][0]:robot_init_pose_range[0][1]:0.1,
                     robot_init_pose_range[1][0]:robot_init_pose_range[1][1]:0.1]
    for x in range(init_x.shape[0]):
        for y in range(init_y.shape[1]):
            robot_init_pose_list.append([init_x[x, y], init_y[x, y]])
    env_range = ((2, 14), (-14, -2))
    poly_raw_list = [[(2, -14), (14, -14), (14, -2), (2, -2)],
                     [(5, -4.5), (5, -4), (11, -4), (11, -4.5)],
                     [(5, -11.5), (5, -12), (11, -12), (11, -11.5)],
                     [(5.25, -6), (4.75, -6), (4.75, -10), (5.25, -10)],
                     [(10.75, -6), (11.25, -6), (11.25, -10), (10.75, -10)]]
    goal_raw_list = [[(2, -14), (14, -14), (14, -2), (2, -2)],
                     [(10.75, -11.5), (10.75, -4), (5.25, -4), (5.25, -11.5)],
                     [(5, -4.5), (5, -4), (11, -4), (11, -4.5)],
                     [(5, -11.5), (5, -12), (11, -12), (11, -11.5)],
                     [(5.25, -6), (4.75, -6), (4.75, -10), (5.25, -10)],
                     [(10.75, -6), (11.25, -6), (11.25, -10), (10.75, -10)],
                     [(6, -4.5), (6, -2), (10, -2), (10, -4.5)],
                     [(6, -11.5), (6, -14), (10, -14), (10, -11.5)],
                     [(5.25, -7), (2, -7), (2, -9), (5.25, -9)],
                     [(10.75, -7), (14, -7), (14, -9), (10.75, -9)]]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)

    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        robot_init_pose = random.choice(robot_init_pose_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(robot_init_pose_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list


def gen_rand_list_env3(size, robot_goal_diff=1.0):

    robot_init_pose_range = ((-9.75, -6.25), (-9, -7))
    robot_init_pose_list = []
    init_x, init_y = np.mgrid[robot_init_pose_range[0][0]:robot_init_pose_range[0][1]:0.1,
                     robot_init_pose_range[1][0]:robot_init_pose_range[1][1]:0.1]
    for x in range(init_x.shape[0]):
        for y in range(init_y.shape[1]):
            robot_init_pose_list.append([init_x[x, y], init_y[x, y]])
    env_range = ((-14, -2), (-14, -2))
    poly_raw_list = [[(-2, -14), (-14, -14), (-14, -2), (-2, -2)],
                     [(-6.5, -6.5), (-6.5, -6), (-9.5, -6), (-9.5, -6.5)],
                     [(-6.5, -9.5), (-6.5, -10), (-9.5, -10), (-9.5, -9.5)],
                     [(-5.25, -6.5), (-4.75, -6.5), (-4.75, -9.5), (-5.25, -9.5)],
                     [(-10.75, -6.5), (-11.25, -6.5), (-11.25, -9.5), (-10.75, -9.5)]]
    goal_raw_list = [[(-2, -14), (-14, -14), (-14, -2), (-2, -2)],
                     [(-5.25, -9.5), (-5.25, -6.5), (-10.75, -6.5), (-10.75, -9.5)],
                     [(-9.5, -6.5), (-9.5, -2), (-14, -2), (-14, -6.5)],
                     [(-6.5, -6.5), (-2, -6.5), (-2, -2), (-6.5, -2)],
                     [(-6.5, -9.5), (-6.5, -14), (-2, -14), (-2, -9.5)],
                     [(-9.5, -9.5), (-14, -9.5), (-14, -14), (-9.5, -14)],
                     [(-6.5, -6.5), (-6.5, -6), (-9.5, -6), (-9.5, -6.5)],
                     [(-6.5, -9.5), (-6.5, -10), (-9.5, -10), (-9.5, -9.5)],
                     [(-5.25, -6.5), (-4.75, -6.5), (-4.75, -9.5), (-5.25, -9.5)],
                     [(-10.75, -6.5), (-11.25, -6.5), (-11.25, -9.5), (-10.75, -9.5)]]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)

    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        robot_init_pose = random.choice(robot_init_pose_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(robot_init_pose_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list


def gen_rand_list_env4(size, robot_goal_diff=5.0):

    env_range = ((-14, -2), (2, 14))
    poly_raw_list = [[(-2, 14), (-14, 14), (-14, 2), (-2, 2)],
                     [(-6, 9), (-6, 9.5), (-9, 9.5), (-9, 9)],
                     [(-6, 7), (-6, 6.5), (-10, 6.5), (-10, 7)],
                     [(-4, 11.5), (-4, 12), (-8, 12), (-8, 11.5)],
                     [(-11.5, 9), (-11.5, 12), (-12, 12), (-12, 9)],
                     [(-3.75, 4), (-3.75, 7), (-4.25, 7), (-4.25, 4)],
                     [(-8, 4), (-8, 4.5), (-12, 4.5), (-12, 4)]]
    goal_raw_list = [[(-2, 14), (-14, 14), (-14, 2), (-2, 2)],
                     [(-6, 9), (-6, 9.5), (-9, 9.5), (-9, 9)],
                     [(-6, 7), (-6, 6.5), (-10, 6.5), (-10, 7)],
                     [(-4, 11.5), (-4, 12), (-8, 12), (-8, 11.5)],
                     [(-11.5, 9), (-11.5, 12), (-12, 12), (-12, 9)],
                     [(-3.75, 4), (-3.75, 7), (-4.25, 7), (-4.25, 4)],
                     [(-8, 4), (-8, 4.5), (-12, 4.5), (-12, 4)]]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)

    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        robot_init_pose = random.choice(goal_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(goal_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list


def gen_goal_position_list(poly_list, env_size=((-6, 6), (-6, 6)), obs_near_th=0.5, sample_step=0.1):

    goal_pos_list = []
    x_pos, y_pos = np.mgrid[env_size[0][0]:env_size[0][1]:sample_step, env_size[1][0]:env_size[1][1]:sample_step]
    for x in range(x_pos.shape[0]):
        for y in range(x_pos.shape[1]):
            tmp_pos = [x_pos[x, y], y_pos[x, y]]
            tmp_point = Point(tmp_pos[0], tmp_pos[1])
            near_obstacle = False
            for poly in poly_list:
                tmp_dis = tmp_point.distance(poly)
                if tmp_dis < obs_near_th:
                    near_obstacle = True
            if near_obstacle is False:
                goal_pos_list.append(tmp_pos)
    return goal_pos_list


def gen_polygon_exterior_list(poly_point_list):

    poly_list = []
    for i, points in enumerate(poly_point_list, 0):
        tmp_poly = Polygon(points)
        if i > 0:
            poly_list.append(tmp_poly)
        else:
            poly_list.append(tmp_poly.exterior)
    return poly_list



def wheeled_network_2_robot_action_decoder(action, wheel_max, wheel_min, diff=0.25):

    l_spd = action[0] * (wheel_max - wheel_min) + wheel_min
    r_spd = action[1] * (wheel_max - wheel_min) + wheel_min
    linear = (l_spd + r_spd) / 2
    angular = (r_spd - l_spd) / diff
    return [linear, angular]


def robot_2_goal_dis_dir(robot_pose, goal_pos):

    delta_x = goal_pos[0] - robot_pose[0]
    delta_y = goal_pos[1] - robot_pose[1]
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
    ego_direction = math.atan2(delta_y, delta_x)
    robot_direction = robot_pose[2]
    while robot_direction < 0:
        robot_direction += 2 * math.pi
    while robot_direction > 2 * math.pi:
        robot_direction -= 2 * math.pi
    while ego_direction < 0:
        ego_direction += 2 * math.pi
    while ego_direction > 2 * math.pi:
        ego_direction -= 2 * math.pi
    pos_dir = abs(ego_direction - robot_direction)
    neg_dir = 2 * math.pi - abs(ego_direction - robot_direction)
    if pos_dir <= neg_dir:
        direction = math.copysign(pos_dir, ego_direction - robot_direction)
    else:
        direction = math.copysign(neg_dir, -(ego_direction - robot_direction))
    return distance, direction



def ddpg_state_rescale(state, state_num,
                       goal_dir_range=math.pi, linear_spd_range=0.5, angular_spd_range=2.0):

    rescale_state_value = [0 for _ in range(state_num)]
    if state[0] > 0:
        rescale_state_value[0] = state[0] / goal_dir_range
    else:
        rescale_state_value[0] = -abs(state[0]) / goal_dir_range
    rescale_state_value[1] = state[1]
    rescale_state_value[2] = state[2] / linear_spd_range
    if state[3] > 0:
        rescale_state_value[3] = state[3] / angular_spd_range
    else:
        rescale_state_value[3] = -abs(state[3]) / angular_spd_range
    rescale_state_value[4:] = state[4:]
    return rescale_state_value

