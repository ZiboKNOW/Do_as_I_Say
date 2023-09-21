#!/usr/bin/env python3
import rospy
import tf2_ros
import numpy as np
import math
import threading
from numpy.linalg import inv
from nav_msgs.msg import Odometry
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import SetMode
from pyquaternion import Quaternion
import time
def cb(ego_position):
    print(ego_position)
ego_position_sub = rospy.Subscriber("/mavros/global_position/local", Odometry, cb)
time.sleep(5)
rospy.set_param("X",0.1)
rospy.set_param("Y",-0.1)
rospy.set_param("Z",2.0)
    