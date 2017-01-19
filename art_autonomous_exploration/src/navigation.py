#!/usr/bin/env python

import rospy
import math
import time
import numpy as np
from robot_perception import RobotPerception
from target_selection import TargetSelection
from path_planning import PathPlanning
from utilities import RvizHandler
from utilities import Print
from utilities import TimerThread

import ipdb
from bresenham import bresenham

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# Class for implementing the navigation module of the robot
class Navigation:

    # Constructor
    def __init__(self):

        # Initializations
        self.robot_perception = RobotPerception()
        self.path_planning = PathPlanning()

        # Check if the robot moves with target or just wanders
        self.move_with_target = rospy.get_param("calculate_target")

        # Flag to check if the vehicle has a target or not
        self.target_exists = False
        self.select_another_target = 0
        self.inner_target_exists = False

        # Container for the current path
        self.path = []
        # Container for the subgoals in the path
        self.subtargets = []

        # Container for the next subtarget. Holds the index of the next subtarget
        self.next_subtarget = 0

        self.time_limit = 20 # seconds

        # Check if subgoal is reached via a timer callback
        rospy.Timer(rospy.Duration(0.10), self.checkTarget)

        # Read the target function
        self.target_selector = rospy.get_param("target_selector")
        print "The selected target function is " + self.target_selector
        self.target_selection = TargetSelection(self.target_selector)

        # ROS Publisher for the path
        self.path_publisher = \
            rospy.Publisher(rospy.get_param('path_pub_topic'), \
            Path, queue_size = 10)

        # ROS Publisher for the subtargets
        self.subtargets_publisher = \
            rospy.Publisher(rospy.get_param('subgoals_pub_topic'),\
            MarkerArray, queue_size = 10)

        # ROS Publisher for the current target
        self.current_target_publisher = \
            rospy.Publisher(rospy.get_param('curr_target_pub_topic'),\
            Marker, queue_size = 10)

        # Timer Thread (initialization)
        self.timerThread = TimerThread(self.time_limit)

    def checkTarget(self, event):
        # Check if we have a target or if the robot just wanders
        if self.inner_target_exists == False or self.move_with_target == False or\
                self.next_subtarget == len(self.subtargets):
          return

        # Check if timer has expired
        if self.timerThread.expired == True:
          Print.art_print('\n~~~~ Time reset ~~~~',Print.RED)
          self.inner_target_exists = False
          self.target_exists = False
          return

        # Get the robot pose in pixels
        [rx, ry] = [\
            self.robot_perception.robot_pose['x_px'] - \
                    self.robot_perception.origin['x'] / self.robot_perception.resolution,\
            self.robot_perception.robot_pose['y_px'] - \
                    self.robot_perception.origin['y'] / self.robot_perception.resolution\
                    ]

        theta_robot = self.robot_perception.robot_pose['th']

        # Clear achieved targets
        self.subtargets = self.subtargets[self.next_subtarget:]
        self.next_subtarget = 0

        # Find the distance between the robot pose and all remaining subtargets
        dx = [rx - st[0] for st in self.subtargets]
        dy = [ry - st[1] for st in self.subtargets]
        dist = [math.hypot(v[0], v[1]) for v in zip(dx, dy)]

        # Try to optimize path only when there is more than one target available
        if len(self.subtargets) > 1:
          ogm = self.robot_perception.getMap()

          theta_goals = np.arctan2(-np.array(dy),-np.array(dx))
          dtheta = theta_goals - theta_robot
          dtheta[dtheta > np.pi] -= 2*np.pi
          dtheta[dtheta < -np.pi] += 2*np.pi

          # Check if there are obstacles between robot and subtarget
          obstacles_subtarget = []
          for st,i in zip(self.subtargets, range(len(self.subtargets))):
            # Find line in pixels between robot and goal
            line_pxls = list(bresenham(int(round(st[0])), int(round(st[1])),\
                                       int(round(rx)), int(round(ry))))
            # Find if there are any obstacles in line
            ogm_line = list(map(lambda pxl: ogm[pxl[0], pxl[1]], line_pxls))
            N_occupied = len(list(filter(lambda x: x > 80, ogm_line)))
            obstacles_subtarget.append(N_occupied)

          # Check if one of next goals is aligned with the current robot theta
          # In this case plan immediately for this goal in order to improve motion behaviour
          for st, i in zip(self.subtargets, range(len(self.subtargets))):
            if np.fabs(dtheta[i]) < np.pi/10 and obstacles_subtarget[i] == 0:
              self.next_subtarget = i
              # Keep resetting  timer, while moving towards target (without any obstacles inbetween)
              self.timerThread.reset()

        # Check if any target is in distance
        min_dist, min_idx = min(zip(dist, range(len(dist))))

        if min_dist < 5:
          self.next_subtarget = min_idx + 1
          # Reset timer if a goal has been reached
          self.timerThread.reset()
          # Check if the final subtarget has been approached
          if self.next_subtarget >= len(self.subtargets):
            self.target_exists = False

        # Publish the current target
        if self.next_subtarget >= len(self.subtargets):
            return

        subtarget = [\
            self.subtargets[self.next_subtarget][0]\
                * self.robot_perception.resolution + \
                self.robot_perception.origin['x'],
            self.subtargets[self.next_subtarget][1]\
                * self.robot_perception.resolution + \
                self.robot_perception.origin['y']\
            ]

        RvizHandler.printMarker(\
            [subtarget],\
            1, # Type: Arrow
            0, # Action: Add
            "map", # Frame
            "art_next_subtarget", # Namespace
            [0, 0, 0.8, 0.8], # Color RGBA
            0.2 # Scale
        )

    # Function that selects the next target, produces the path and updates
    # the coverage field. This is called from the speeds assignment code, since
    # it contains timer callbacks
    def selectTarget(self):

        # Cancel previous goal timer
        self.timerThread.stop()

        # IMPORTANT: The robot must be stopped if you call this function until
        # it is over
        # Check if we have a map
        while self.robot_perception.have_map == False:
          Print.art_print("Navigation: No map yet", Print.RED)
          return

        print "\nClearing all markers"
        RvizHandler.printMarker(\
            [[0, 0]],\
            1, # Type: Arrow
            3, # Action: delete all
            "map", # Frame
            "null", # Namespace
            [0,0,0,0], # Color RGBA
            0.1 # Scale
        )

        print '\n\n----------------------------------------------------------'
        print "Navigation: Producing new target"
        # We are good to continue the exploration
        # Make this true in order not to call it again from the speeds assignment
        self.target_exists = True

        # Gets copies of the map and coverage
        local_ogm = self.robot_perception.getMap()
        local_ros_ogm = self.robot_perception.getRosMap()
        local_coverage = self.robot_perception.getCoverage()
        print "Got the map and Coverage"
        self.path_planning.setMap(local_ros_ogm)

        # Once the target has been found, find the path to it
        # Get the global robot pose
        g_robot_pose = self.robot_perception.getGlobalCoordinates(\
              [self.robot_perception.robot_pose['x_px'],\
              self.robot_perception.robot_pose['y_px']])

        # Call the target selection function to select the next best goal
        # Choose target function
        self.path = []
        force_random = False
        while len(self.path) == 0:
          start = time.time()
          target = self.target_selection.selectTarget(\
                    local_ogm,\
                    local_coverage,\
                    self.robot_perception.robot_pose,
                    self.robot_perception.origin,
                    self.robot_perception.resolution,
                    force_random)
          self.path = self.path_planning.createPath(\
              g_robot_pose,\
              target,
              self.robot_perception.resolution)
          print "Navigation: Path for target found with " + str(len(self.path)) +\
              " points"
          if len(self.path) == 0:
            Print.art_print(\
                "Path planning failed. Fallback to random target selection", \
                Print.RED)
            force_random = True

        # Reverse the path to start from the robot
        self.path = self.path[::-1]

        # Break the path to subgoals every 2 pixels (1m = 20px)
        step = 1
        n_subgoals = (int) (len(self.path)/step)
        self.subtargets = []
        for i in range(0, n_subgoals):
          self.subtargets.append(self.path[i * step])
        self.subtargets.append(self.path[-1])
        self.next_subtarget = 0
        print "The path produced " + str(len(self.subtargets)) + " subgoals"

        ######################### NOTE: QUESTION  ##############################
        # The path is produced by an A* algorithm. This means that it is
        # optimal in length but 1) not smooth and 2) length optimality
        # may not be desired for coverage-based exploration
        ########################################################################

        # Start timer thread
        self.timerThread.start()

        # Publish the path for visualization purposes
        ros_path = Path()
        ros_path.header.frame_id = "map"
        for p in self.path:
          ps = PoseStamped()
          ps.header.frame_id = "map"
          ps.pose.position.x = 0
          ps.pose.position.y = 0

          ######################### NOTE: QUESTION  ##############################
          # Fill the ps.pose.position values to show the path in RViz
          # You must understand what self.robot_perception.resolution
          # and self.robot_perception.origin are.
          # ipdb.set_trace()
          ps.pose.position.x = p[0] * self.robot_perception.resolution \
              + self.robot_perception.origin['x']
          ps.pose.position.y = p[1] * self.robot_perception.resolution \
              + self.robot_perception.origin['y']
          ########################################################################

          ros_path.poses.append(ps)
        self.path_publisher.publish(ros_path)

        # Publish the subtargets for visualization purposes
        subtargets_mark = []
        for s in self.subtargets:
          subt = [
            s[0] * self.robot_perception.resolution + \
                    self.robot_perception.origin['x'],
            s[1] * self.robot_perception.resolution + \
                    self.robot_perception.origin['y']
            ]
          subtargets_mark.append(subt)

        RvizHandler.printMarker(\
            subtargets_mark,\
            2, # Type: Sphere
            0, # Action: Add
            "map", # Frame
            "art_subtargets", # Namespace
            [0, 0.8, 0.0, 0.8], # Color RGBA
            0.2 # Scale
        )

        self.inner_target_exists = True

    def velocitiesToNextSubtarget(self):

        [linear, angular] = [0, 0]

        [rx, ry] = [\
            self.robot_perception.robot_pose['x_px'] - \
                    self.robot_perception.origin['x'] / self.robot_perception.resolution,\
            self.robot_perception.robot_pose['y_px'] - \
                    self.robot_perception.origin['y'] / self.robot_perception.resolution\
                    ]
        theta = self.robot_perception.robot_pose['th']

        ######################### NOTE: QUESTION  ##############################
        # The velocities of the robot regarding the next subtarget should be
        # computed. The known parameters are the robot pose [x,y,th] from
        # robot_perception and the next_subtarget [x,y]. From these, you can
        # compute the robot velocities for the vehicle to approach the target.
        # Hint: Trigonometry is required
        l_max = 0.3
        a_max = 0.3

        if self.subtargets and self.next_subtarget <= len(self.subtargets) - 1:
          st_x = self.subtargets[self.next_subtarget][0]
          st_y = self.subtargets[self.next_subtarget][1]

          theta_rg = np.arctan2(st_y - ry, st_x - rx)
          dtheta = (theta_rg - theta)
          if dtheta > np.pi:
            omega = (dtheta - 2*np.pi)/np.pi
          elif dtheta < -np.pi:
            omega = (dtheta + 2*np.pi)/np.pi
          else:
            omega = (theta_rg - theta)/np.pi

          # Nonlinear relations derived from experimentation
          linear = l_max * ((1 - np.abs(omega)) ** 5)
          angular = a_max * np.sign(omega) * (abs(omega) ** (1/5))
        ######################### NOTE: QUESTION  ##############################

        return [linear, angular]
