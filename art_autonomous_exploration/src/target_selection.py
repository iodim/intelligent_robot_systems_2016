#!/usr/bin/env python

import rospy
import random
import math
import time
import numpy as np
from timeit import default_timer as timer
from utilities import RvizHandler
from utilities import OgmOperations
from utilities import Print
from brushfires import Brushfires
from topology import Topology
import scipy
from path_planning import PathPlanning

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


    def selectTarget(self, init_ogm, ros_ogm, coverage, robot_pose, origin, \
        resolution, force_random = False):

        target = [-1, -1]

        ######################### NOTE: QUESTION  ##############################
        # Implement a smart way to select the next target. You have the
        # following tools: ogm_limits, Brushfire field, OGM skeleton,
        # topological nodes.

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

        self.path_planning.setMap(ros_ogm)
        g_robot_pose = [robot_pose['x_px'] - int(origin['x'] / resolution),\
                        robot_pose['y_px'] - int(origin['y'] / resolution)]

        # Remove all covered nodes from the candidate list
        nodes = np.array(nodes)
        uncovered_idxs = np.array([coverage[n[0], n[1]] == 0 for n in nodes])
        goals = nodes[uncovered_idxs]

        w_dist = np.full(len(goals), np.inf)
        w_turn = np.full(len(goals), np.inf)
        w_cove = np.full(len(goals), np.inf)
        w_topo = np.full(len(goals), np.inf)

        for idx, node in zip(range(len(goals)), goals):
          # ipdb.set_trace()
          subgoals = np.array(self.path_planning.createPath(g_robot_pose, node, resolution))

          # If the path is impossible (empty subgoals) then skip to the next iteration
          if subgoals.size == 0:
            continue

          # subgoals should contain the robot pose, so we don't need to diff it explicitly
          subgoal_vectors = np.diff(subgoals, axis=0)
          # ipdb.set_trace()

          # # Distance cost
          dists = [math.hypot(v[0], v[1]) for v in subgoal_vectors]
          w_dist[idx] = np.sum(dists)

          # # Turning cost
          # w_turn[idx] = 0
          # for v, w in zip(subgoal_vectors[:-1], subgoal_vectors[1:]):
          #   c = np.dot(v,w) / np.norm(w) / np.norm(w)
          #   w_turn[idx] += np.arccos(np.clip(c, -1, 1))

          # # Coverage cost

        # ipdb.set_trace()
        min_dist, min_idx = min(zip(w_dist, range(len(w_dist))))
        target = nodes[min_idx]

        # Random point
        if self.method == 'random' or force_random == True:
          target = self.selectRandomTarget(ogm, coverage, brush, ogm_limits)
        ########################################################################

        return target

    # def calcTopologicalCost(self, ogm, node):

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

