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
        cov_dx = scipy.ndimage.sobel(coverage, 0)
        cov_dy = scipy.ndimage.sobel(coverage, 1)
        cov_frontier = np.hypot(cov_dx, cov_dy)
        cov_frontier *= 100 / np.max(cov_frontier)
        cov_frontier = 100 * (cov_frontier > 80)

        # Remove the edges that correspond to obstacles instead of frontiers (in a 5x5 radius)
        kern = 5
        i_rng = np.matlib.repmat(np.arange(-(kern/2), kern/2 + 1).reshape(kern, 1), 1, kern)
        j_rng = np.matlib.repmat(np.arange(-(kern/2), kern/2 + 1), kern, 1)
        for i in range((kern/2), cov_frontier.shape[0] - (kern/2)):
          for j in range((kern/2) , cov_frontier.shape[1] - (kern/2)):
            if cov_frontier[i, j] == 100:
              if np.any(ogm[i + i_rng, j + j_rng] > 99):
                cov_frontier[i, j] = 0

        # Save coverage frontier as image (for debugging purposes)
        # scipy.misc.imsave('test.png', np.rot90(cov_frontier))

        # Frontier detection/grouping
        labeled_frontiers, num_frontiers = scipy.ndimage.label(cov_frontier, np.ones((3, 3)))

        # Calculate the centroid and its cost, for each frontier
        goals = np.full((num_frontiers, 2), -1)
        w_dist = np.full(len(goals), -1)
        w_turn = np.full(len(goals), -1)
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
          goals[i - 1, :] = np.array([points[0][nearest_idx], points[1][nearest_idx]])

          # Save centroids for later visualisation (for debugging purposes)
          # labeled_frontiers[int(goals[i - 1, 0]) + i_rng, int(goals[i - 1, 1]) + j_rng] = i

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

        # Save frontier groupings as an image (for debugging purposes)
        # cmap = plt.cm.jet
        # norm = plt.Normalize(vmin=labeled_frontiers.min(), vmax=labeled_frontiers.max())
        # image = cmap(norm(labeled_frontiers))
        # plt.imsave('erma.png', np.rot90(image))

        # Remove invalid goals and weights
        valids = w_dist != -1
        goals = goals[valids]
        w_dist = w_dist[valids]
        w_turn = w_turn[valids]

        # Normalize weights
        w_dist = (w_dist - min(w_dist))/(max(w_dist) - min(w_dist))
        w_turn = (w_turn - min(w_turn))/(max(w_turn) - min(w_turn))

        # Goal cost function
        c_dist = 1
        c_turn = 2
        costs = c_dist * w_dist + c_turn * w_turn

        # min_dist, min_idx = min(zip(costs, range(len(costs))))
        min_idx = costs.argmin()

        Print.art_print("Target selection time: " + str(time.time() - tinit), Print.ORANGE)

        return goals[min_idx]

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

