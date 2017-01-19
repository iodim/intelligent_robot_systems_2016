#!/usr/bin/env python

import rospy
import random
import math
import time
import scipy
import numpy as np
from timeit import default_timer as timer
from utilities import RvizHandler
from utilities import OgmOperations
from utilities import Print
from brushfires import Brushfires
from topology import Topology
from path_planning import PathPlanning

import matplotlib.pyplot as plt
from bresenham import bresenham
import ipdb

# Class for selecting the next best target
class TargetSelection:

    # Constructor
    def __init__(self, selection_method):
        self.goals_position = []
        self.goals_value = []
        self.omega = 0.0
        self.radius = 0
        self.method = selection_method

        self.brush = Brushfires()
        self.topo = Topology()
        self.path_planning = PathPlanning()


    def selectTarget(self, ogm, coverage, robot_pose, origin, \
        resolution, force_random = False):

        # Random point
        if self.method == 'random' or force_random == True:

          init_ogm = ogm

          # Find only the useful boundaries of OGM. Only there calculations
          # have meaning
          ogm_limits = OgmOperations.findUsefulBoundaries(init_ogm, origin, resolution)

          # Blur the OGM to erase discontinuities due to laser rays
          ogm = OgmOperations.blurUnoccupiedOgm(init_ogm, ogm_limits)

          # Calculate Brushfire field
          tinit = time.time()
          brush = self.brush.obstaclesBrushfireCffi(ogm, ogm_limits)
          Print.art_print("Brush time: " + str(time.time() - tinit), Print.ORANGE)

          # Calculate skeletonization
          tinit = time.time()
          skeleton = self.topo.skeletonizationCffi(ogm, \
                     origin, resolution, ogm_limits)
          Print.art_print("Skeletonization time: " + str(time.time() - tinit), Print.ORANGE)

          # Find topological graph
          tinit = time.time()
          nodes = self.topo.topologicalNodes(ogm, skeleton, coverage, origin, \
                  resolution, brush, ogm_limits)
          Print.art_print("Topo nodes time: " + str(time.time() - tinit), Print.ORANGE)

          # Visualization of topological nodes
          vis_nodes = []
          for n in nodes:
              vis_nodes.append([
                  n[0] * resolution + origin['x'],
                  n[1] * resolution + origin['y']
              ])
          RvizHandler.printMarker(\
              vis_nodes,\
              1, # Type: Arrow
              0, # Action: Add
              "map", # Frame
              "art_topological_nodes", # Namespace
              [0.3, 0.4, 0.7, 0.5], # Color RGBA
              0.1 # Scale
          )

          target = self.selectRandomTarget(ogm, coverage, brush, ogm_limits)
          return target

        tinit = time.time()

        g_robot_pose = [robot_pose['x_px'] - int(origin['x'] / resolution), \
                        robot_pose['y_px'] - int(origin['y'] / resolution)]

        # Calculate coverage frontier with sobel filters
        tinit = time.time()
        cov_dx = scipy.ndimage.sobel(coverage, 0)
        cov_dy = scipy.ndimage.sobel(coverage, 1)
        cov_frontier = np.hypot(cov_dx, cov_dy)
        cov_frontier *= 100 / np.max(cov_frontier)
        cov_frontier = 100 * (cov_frontier > 80)
        Print.art_print("Sobel filters time: " + str(time.time() - tinit), Print.BLUE)

        # Remove the edges that correspond to obstacles instead of frontiers (in a 5x5 radius)
        kern = 5
        tinit = time.time()
        i_rng = np.matlib.repmat(np.arange(-(kern/2), kern/2 + 1).reshape(kern, 1), 1, kern)
        j_rng = np.matlib.repmat(np.arange(-(kern/2), kern/2 + 1), kern, 1)
        for i in range((kern/2), cov_frontier.shape[0] - (kern/2)):
          for j in range((kern/2) , cov_frontier.shape[1] - (kern/2)):
            if cov_frontier[i, j] == 100:
              if np.any(ogm[i + i_rng, j + j_rng] > 99):
                cov_frontier[i, j] = 0
        Print.art_print("Frontier trimming time: " + str(time.time() - tinit), Print.BLUE)

        # Save coverage frontier as image (for debugging purposes)
        # scipy.misc.imsave('test.png', np.rot90(cov_frontier))

        # Frontier detection/grouping
        tinit = time.time()
        labeled_frontiers, num_frontiers = scipy.ndimage.label(cov_frontier, np.ones((3, 3)))
        Print.art_print("Frontier grouping time: " + str(time.time() - tinit), Print.BLUE)

        goals = np.full((num_frontiers, 2), -1)
        w_dist = np.full(len(goals), -1)
        w_turn = np.full(len(goals), -1)
        w_size = np.full(len(goals), -1)
        w_obst = np.full(len(goals), -1)

        # Calculate the centroid and its cost, for each frontier
        for i in range(1, num_frontiers + 1):
          points = np.where(labeled_frontiers == i)

          # Discard small groupings (we chose 20 as a treshold arbitrarily)
          group_length = points[0].size
          if group_length < 20:
            labeled_frontiers[points] = 0
            continue
          sum_x = np.sum(points[0])
          sum_y = np.sum(points[1])
          centroid = np.array([sum_x/group_length, sum_y/group_length]).reshape(2, 1)

          # Find the point on the frontier nearest (2-norm) to the centroid, and use it as goal
          nearest_idx = np.linalg.norm(np.array(points) - centroid, axis=0).argmin()
          print ogm[int(points[0][nearest_idx]), int(points[1][nearest_idx])]
          goals[i - 1, :] = np.array([points[0][nearest_idx], points[1][nearest_idx]])

          # Save centroids for later visualisation (for debugging purposes)
          labeled_frontiers[int(goals[i - 1, 0]) + i_rng, int(goals[i - 1, 1]) + j_rng] = i

          # Calculate size of obstacles between robot and goal
          line_pxls = list(bresenham(int(goals[i-1,0]), int(goals[i-1,1]),\
                                     g_robot_pose[0], g_robot_pose[1]))

          ogm_line = list(map(lambda pxl: ogm[pxl[0],pxl[1]],line_pxls))

          N_occupied = len(list(filter(lambda x: x>25, ogm_line)))
          N_line     = len(line_pxls)
          w_obst[i-1] = float(N_occupied)/N_line
          # print('Occupied  = '+str(N_occupied))
          # print('Full Line = '+str(N_line))
          # ipdb.set_trace()

          # Manhattan distance
          w_dist[i - 1] = scipy.spatial.distance.cityblock(goals[i - 1, :], g_robot_pose)

          # Missalignment
          theta = np.arctan2(goals[i - 1, 1] - g_robot_pose[1], goals[i - 1, 0] - g_robot_pose[0])
          w_turn[i - 1] = (theta - robot_pose['th'])
          if w_turn[i - 1] > np.pi:
            w_turn[i - 1] -= 2 * np.pi
          elif w_turn[i - 1] < -np.pi:
            w_turn[i - 1] += 2 * np.pi
          # We don't care about the missalignment direction so we abs() it
          w_turn[i - 1] = np.abs(w_turn[i - 1])

          # Frontier size
          w_size[i - 1] = group_length

        # Save frontier groupings as an image (for debugging purposes)
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=labeled_frontiers.min(), vmax=labeled_frontiers.max())
        image = cmap(norm(labeled_frontiers))
        plt.imsave('frontiers.png', np.rot90(image))

        # Remove invalid goals and weights
        valids = w_dist != -1
        goals = goals[valids]
        w_dist = w_dist[valids]
        w_turn = w_turn[valids]
        w_size = w_size[valids]
        w_obst = w_obst[valids]

        # Normalize weights
        w_dist = (w_dist - min(w_dist))/(max(w_dist) - min(w_dist))
        w_turn = (w_turn - min(w_turn))/(max(w_turn) - min(w_turn))
        w_size = (w_size - min(w_size))/(max(w_size) - min(w_size))

        # Goal cost function
        c_dist = 3
        c_turn = 2
        c_size = 1
        c_obst = 4
        costs = c_dist * w_dist + c_turn * w_turn + c_size * w_size+ c_obst * w_obst

        min_idx = costs.argmin()

        Print.art_print("Target selection time: " + str(time.time() - tinit), Print.ORANGE)
        print costs
        print goals

        ## Safety Distance from obstacles 
        # Goal Coordinates
        x_goal = goals[min_idx,0]
        y_goal = goals[min_idx,1]

        # Obstacles in a 40x40 neighborhood
        ogm_segment = ogm[y_goal-20: y_goal+20, y_goal-20: y_goal+20]
        obs_idxs = np.where(ogm_segment > 80)

        # If there are no obstacles terminate normally
        if obs_idxs[0].size == 0:
            return goals[min_idx]


        obs_idxs = np.array([obs_idxs[0],obs_idxs[1]]).transpose()

        # Find closest obstacle
        distances = np.hypot(obs_idxs[0:,0] - 20,obs_idxs[0:,1] - 20)
        closest_idx = distances.argmin()
        min_dist = distances.min()
        closest_obstacle = obs_idxs[closest_idx] - np.array([20,20]) + goals[min_idx]

        # Calculate new goal:
        dist_from_obstacles = 20
        normal_vector = goals[min_idx] - closest_obstacle
        normal_vector = normal_vector/np.hypot(normal_vector[0],normal_vector[1])
        new_goal = closest_obstacle + dist_from_obstacles * normal_vector

        return new_goal.round()

    def selectRandomTarget(self, ogm, coverage, brushogm, ogm_limits):
      # The next target in pixels
        tinit = time.time()
        next_target = [0, 0]
        found = False
        while not found:
          x_rand = random.randint(0, ogm.shape[0] - 1)
          y_rand = random.randint(0, ogm.shape[1] - 1)
          if ogm[x_rand][y_rand] < 50 and coverage[x_rand][y_rand] < 50 and \
              brushogm[x_rand][y_rand] > 5:
            next_target = [x_rand, y_rand]
            found = True
        Print.art_print("Select random target time: " + str(time.time() - tinit), \
            Print.ORANGE)
        return next_target

    def lineOccupation(self,ogm_line):
        occupied = 0
        potentially_occupied = 0
        found_obstacle = False
        for grid in ogm_line:
            if grid > 90:
                occupied_cells += 1 + potentially_occupied
            elif grid < 55 and grid > 45 and found_obstacle:
                potentially_occupied += 1
            elif grid < 20:
                potentially_occupied = 0
                found_obstacle = False
        return occupied



