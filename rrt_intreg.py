from steer import steer
from chk_collision import chk_collision_rrt, map_hull_points
from dist import dist
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
    numNodes = 500 #de schimbat numarul de puncte sa fie mai mic

    q_start = {'coord': [x, y], 'cost': 0, 'parent': 0}
    q_goal = {'coord': [x_goal, y_goal], 'cost': 0}
    poly = []

    nodes = [q_start]
    pathNodes = [q_start]

    img_path = "/home/internship/worldHomeMap.pgm"
    img_yaml = "worldHomeMap.yaml"
    x_min, x_max, y_min, y_max = intervalCoord(img_path, img_yaml)

    map_points = plot_obstacle_poly(ax, "black", poly, img_path)

    hull_points = np.array(map_points)
    hull = ConvexHull(hull_points)
    polygon = Polygon(hull_points[hull.vertices])

    vector_obstacle_hulls = map_hull_points(img_path)
    #print(vector_obstacle_hulls)

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
        
        if chk_collision_rrt([q_near['coord'], q_new['coord']], vector_obstacle_hulls) == 0:
            ax.plot([q_near['coord'][0], q_new['coord'][0]],
                    [q_near['coord'][1], q_new['coord'][1]],
                    'k-', linewidth=1)
            plt.pause(0.001)

            q_new['cost'] = dist(q_new['coord'], q_near['coord']) + q_near['cost']
            q_new['parent'] = nodes.index(q_near)
            nodes.append(q_new)

            if dist(q_new['coord'], q_goal['coord']) < EPS and chk_collision_rrt([q_new['coord'], q_goal['coord']], vector_obstacle_hulls) == 0:
                q_goal['parent'] = nodes.index(q_new)
                nodes.append(q_goal)
                print("Goal reached!")
                break

    path = []
    if q_goal in nodes:
        current = q_goal
    else:
        current = min(nodes, key=lambda n: dist(n['coord'], q_goal['coord']))

    while 'parent' in current:
        path.append(current['coord'])
        current = nodes[current['parent']]
    path.append(q_start['coord'])
    path.reverse()

    for i in range(len(path) - 1):
        ax.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], 'g-', linewidth=2)
        plt.pause(0.001)

    return path

#fig, ax = plt.subplots()
#q = rrt(1, 1, ax, 2, 2)
