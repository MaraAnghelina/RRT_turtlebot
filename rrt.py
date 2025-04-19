from ccw import ccw
from steer import steer
from chk_collision import chk_collision
from dist import dist
from no_collision import no_collision
from slam_to_mat import plot_obstacle_poly
from testImg import process_map
from coordImg import intervalCoord

import numpy as np
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull


def rrt(x, y, ax, x_goal, y_goal):
    EPS = 0.5
    numNodes = 5 #de schimbat numarul de puncte sa fie mai mic

    q_start = {'coord': [x, y], 'cost': 0, 'parent': 0}
    q_goal = {'coord': [x_goal, y_goal], 'cost': 0}
    poly = []

    nodes = [q_start]
    pathNodes = [q_start]

    img_path = "/home/internship/processed_map_inv.pgm"
    img_yaml = "map.yaml"
    x_min, x_max, y_min, y_max = intervalCoord(img_path, img_yaml)

    map_points = plot_obstacle_poly(ax, "black", poly, img_path)

    hull_points = np.array(map_points)
    hull = ConvexHull(hull_points)
    polygon = Polygon(hull_points[hull.vertices])

    ax.plot(x , y, 'go', markersize=5, markerfacecolor='g')

    plt.ion()
    plt.show()

    result1 = []
    elapsed_time1 = []

    for i in range(numNodes):
        K1 = np.random.rand()
        thenorm1 = np.linalg.norm(K1)
        
        while True:
            q_rand_candidate = [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)]
            if polygon.contains(Point(q_rand_candidate)):
                q_rand = q_rand_candidate
                break
        #q_rand = [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)]
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
            r = 30
            for node in nodes:
                if chk_collision([node['coord'], q_new['coord']], poly) == 0 and dist(node['coord'], q_new['coord']) <= r:
                    q_nearest.append(node)
            
            q_min = q_near
            C_min = q_new['cost']
            
            for neighbor in q_nearest:
                if dist(neighbor['coord'], q_new['coord']) + neighbor['cost'] < C_min:
                    q_min = neighbor
                    C_min = neighbor['cost'] + dist(neighbor['coord'], q_new['coord'])
                    #ax.plot([q_min['coord'][0], q_new['coord'][0]], [q_min['coord'][1], q_new['coord'][1]], 'g-')
                    plt.pause(0.01)
            
            q_new['parent'] = nodes.index(q_min)
            nodes.append(q_new)

    q_next = min(nodes, key=lambda p: dist(p['coord'], q_goal['coord']))
    print(q_next)
    return q_next

#fig, ax = plt.subplots()
#q = rrt(1, 1, ax, 2, 2)
