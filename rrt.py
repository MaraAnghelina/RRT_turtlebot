from ccw import ccw
from steer import steer
from chk_collision import chk_collision
from dist import dist
from no_collision import no_collision
from slam_to_mat import plot_obstacle_poly

import numpy as np
import matplotlib.pyplot as plt


def rrt(x, y):
    x_max = 0.51
    y_max = 2.51
    EPS = 1
    numNodes = 500

    q_start = {'coord': [x, y], 'cost': 0, 'parent': 0}
    q_goal = {'coord': [-0.4, 1.94], 'cost': 0}
    poly = []

    nodes = [q_start]

    fig, ax = plt.subplots()

    plot_obstacle_poly(ax, "black", poly)

    print(x_max, y_max)
    ax.plot(x, y, 'go', markersize=5, markerfacecolor='g')
    ax.plot(-0.4, 1.94, 'ro', markersize=5, markerfacecolor='r')

    plt.ion()
    plt.show()

    result1 = []
    elapsed_time1 = []

    for i in range(numNodes):
        K1 = np.random.rand()
        thenorm1 = np.linalg.norm(K1)
        
        q_rand = [np.random.uniform(-0.5, x_max), np.random.uniform(-2.5, y_max)]
        ax.plot(q_rand[0], q_rand[1], 'x', color=[0, 0.4470, 0.7410])
        plt.pause(0.01)
        
        if any(np.array_equal(node['coord'], q_goal['coord']) for node in nodes):
            break
        
        ndist = [dist(node['coord'], q_rand) for node in nodes]
        q_near = nodes[np.argmin(ndist)]
        
        q_new = {'coord': steer(q_rand, q_near['coord'], np.min(ndist), EPS)}
        
        if chk_collision([q_near['coord'], q_new['coord']], poly) == 0:
            ax.plot([q_near['coord'][0], q_new['coord'][0]], [q_near['coord'][1], q_new['coord'][1]], 'k-', linewidth=2)
            plt.pause(0.01)
            
            q_new['cost'] = dist(q_new['coord'], q_near['coord']) + q_near['cost']
            
            q_nearest = []
            r = 60
            for node in nodes:
                if chk_collision([node['coord'], q_new['coord']], poly) == 0 and dist(node['coord'], q_new['coord']) <= r:
                    q_nearest.append(node)
            
            q_min = q_near
            C_min = q_new['cost']
            
            for neighbor in q_nearest:
                if dist(neighbor['coord'], q_new['coord']) + neighbor['cost'] < C_min:
                    q_min = neighbor
                    C_min = neighbor['cost'] + dist(neighbor['coord'], q_new['coord'])
                    ax.plot([q_min['coord'][0], q_new['coord'][0]], [q_min['coord'][1], q_new['coord'][1]], 'g-')
                    plt.pause(0.01)
            
            q_new['parent'] = nodes.index(q_min)
            nodes.append(q_new)
        
        result1.append(thenorm1)
        elapsed_time1.append(K1)

    result_avg1 = np.mean(result1)
    time_avg1 = np.mean(elapsed_time1)

    D = [dist(node['coord'], q_goal['coord']) for node in nodes]

    result2 = []
    elapsed_time2 = []

    for j in range(len(nodes)):
        K2 = np.random.rand()
        thenorm2 = np.linalg.norm(K2)
        result2.append(thenorm2)
        elapsed_time2.append(K2)

    result_avg2 = np.mean(result2)
    time_avg2 = np.mean(elapsed_time2)

    q_final = nodes[np.argmin(D)]
    q_goal['parent'] = nodes.index(q_final)
    q_end = q_goal

    nodes.append(q_goal)

    while q_end['parent'] != 0:
        start = q_end['parent']
        ax.plot([q_end['coord'][0], nodes[start]['coord'][0]], [q_end['coord'][1], nodes[start]['coord'][1]], 'r-', linewidth=3)
        plt.pause(0.01)
        
        q_end = nodes[start]

    plt.ioff()
    plt.show()