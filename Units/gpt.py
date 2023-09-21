#!/usr/bin/env python
# license removed for brevity
import rospy
import tf2_ros
import numpy as np
import math
import threading
from numpy.linalg import inv
from nav_msgs.msg import Odometry
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates
from mavros_msgs.srv import SetMode
from pyquaternion import Quaternion
class ENV_API:

    def __init__(self):
        rospy.init_node('ENV_API', anonymous=True)
        self.name_space = rospy.get_namespace().strip('/')
        # self.object_name = []
        self.object_position ={}
        self.ego_position_sub = rospy.Subscriber("/mavros/global_position/local", Odometry, self.save_ego_position)
        self.obj_position_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.save_object_position)
        self.target_pose_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.rate = rospy.Rate(10)
        print("waiting for the service")
        rospy.wait_for_service('/mavros/set_mode')
        print("Service has started")
        if not self.set_offboard_mode():
            rospy.logerr("Failed to set OFFBOARD mode")
            return
        self.ego_position = Odometry()
        self.got_ego_pose = False
        # take off
        rospy.set_param("X",0)
        rospy.set_param("Y",0.0)
        rospy.set_param("Z",10.0)
        rospy.set_param("z",1.0)
    def run(self):
        rospy.spin()
    def save_ego_position(self, ego_position):
        self.ego_position = ego_position
        self.got_ego_pose = True
    def save_object_position(self, message):
        for name in message.name:
            if name =='law_office':
                object_index = message.name.index(name)
                object_pose = message.pose[object_index]
                self.object_position.update({'law office' : object_pose})
            elif name =='oak_tree':
                object_index = message.name.index(name)
                object_pose = message.pose[object_index]
                self.object_position.update({'green tree' : object_pose})                
            elif name =='salon':
                object_index = message.name.index(name)
                object_pose = message.pose[object_index]
                self.object_position.update({name: object_pose})  
            elif name =='suv':
                object_index = message.name.index(name)
                object_pose = message.pose[object_index]
                self.object_position.update({'black suv': object_pose})
            elif name =='person_standing':
                object_index = message.name.index(name)
                object_pose = message.pose[object_index]
                self.object_position.update({'standing person in white': object_pose})
            elif name =='person_walking':
                object_index = message.name.index(name)
                object_pose = message.pose[object_index]
                self.object_position.update({'walking person in white': object_pose})

        # self.object_position.update({'object name':obj_position})
    def get_obj_pos(self, name):
        pos = self.object_position[name]
        print('global pose: ',pos)
        orientation = self.ego_position.pose.pose.orientation
        position = self.ego_position.pose.pose.position
        roll, pitch, yaw = self.quaternion_to_euler([orientation.x,orientation.y,orientation.z,orientation.w])
        # print('roll: ',roll,' pitch: ',pitch,'yaw: ',yaw)
        Quaternion_filter = self.euler_to_quaternion(0, 0, yaw)
        # print('after change: ',Quaternion_filter)
        local2world = Quaternion(Quaternion_filter[0],Quaternion_filter[1],Quaternion_filter[2],Quaternion_filter[3])
        world2local_rotation = inv(local2world.rotation_matrix)
        world_to_local_trans_mat = self.get_trans_mat_world2local(position,world2local_rotation)
        global_pose = [pos.position.x, pos.position.y, pos.position.z, 1]
        local_pose = world_to_local_trans_mat @ global_pose
        return  [local_pose[0], local_pose[1], local_pose[2], 0, 0, 0]
    def get_ego_pos(self):
        return [0,0,0,0,0,0]

    def get_obj_names(self):
        return ['law office', 'green tree', 'salon', 'black suv', 'standing person in white', 'walking person in white']

    def is_obj_visible(self,name):
        return True or False
    def move_to_pos(self,target_position):
        print('target pose: ',target_position)
        target_pose = PoseStamped()
        target_pose.header.stamp = rospy.Time.now()
        orientation = self.ego_position.pose.pose.orientation
        position = self.ego_position.pose.pose.position
        roll, pitch, yaw = self.quaternion_to_euler([orientation.x,orientation.y,orientation.z,orientation.w])
        # print('roll: ',roll,' pitch: ',pitch,'yaw: ',yaw)
        print('origin yaw: ',yaw)
        yaw += target_position[3]
        print('changed yaw: ',yaw)
        Quaternion_filter = self.euler_to_quaternion(0, 0, yaw)
        # print('after change: ',Quaternion_filter)
        local2world = Quaternion(Quaternion_filter[0],Quaternion_filter[1],Quaternion_filter[2],Quaternion_filter[3])
        world2local_rotation = inv(local2world.rotation_matrix)
        local_to_world_trans_mat = inv(self.get_trans_mat_world2local(position,world2local_rotation))
        target_point_local = np.append(target_position[:3], 1) 
        target_pose_world = local_to_world_trans_mat @ target_point_local
        target_pose.pose.position.x = float(target_pose_world[0])
        target_pose.pose.position.y = float(target_pose_world[1])
        target_pose.pose.position.z = float(target_pose_world[2])
        target_pose.pose.orientation.x = float(Quaternion_filter[0])
        target_pose.pose.orientation.y = float(Quaternion_filter[1])
        target_pose.pose.orientation.z = float(Quaternion_filter[2])
        target_pose.pose.orientation.w = float(Quaternion_filter[3])
        print('global target pose: ', target_pose.pose.position)
        rospy.set_param("X",target_pose.pose.position.x)
        rospy.set_param("Y",target_pose.pose.position.y)
        rospy.set_param("Z",target_pose.pose.position.z)
        rospy.set_param("x",target_pose.pose.orientation.x)
        rospy.set_param("y",target_pose.pose.orientation.y)
        rospy.set_param("z",target_pose.pose.orientation.z)
        rospy.set_param("w",target_pose.pose.orientation.w)
        # print(target_pose.pose.position.x)
        # print(target_pose.pose.position.y)
        # print(target_pose.pose.position.z)
        # rospy.set_param("X",0.02)
        # rospy.set_param("Y",0.02)
        # rospy.set_param("Z",2)

    def get_trans_mat_world2local(self, drone_pose, rotation):
        im_position = [drone_pose.x,drone_pose.y, drone_pose.z]
        im_position = np.array(im_position).reshape((3, 1))
        extrinsic_mat = np.hstack((rotation, - rotation @ im_position))
        norm = np.array([0,0,0,1]).reshape((1,4))
        project_mat = np.vstack((extrinsic_mat, norm))
        return project_mat
    
    def set_offboard_mode(self):
        mode_resp = self.set_mode_client(custom_mode="OFFBOARD")
        return mode_resp.mode_sent
    
    def set_auto_mode(self):
        mode_resp = self.set_mode_client(custom_mode="AUTO")
        return mode_resp.mode_sent

    def quaternion_to_euler(self, quaternion):
    # 归一化四元数
        q0, q1, q2, q3 = quaternion
        norm = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q0 /= norm
        q1 /= norm
        q2 /= norm
        q3 /= norm

        # 计算旋转矩阵的元素
        r11 = 2*(q0*q0 + q1*q1) - 1
        r12 = 2*(q1*q2 - q0*q3)
        r13 = 2*(q1*q3 + q0*q2)
        r21 = 2*(q1*q2 + q0*q3)
        r22 = 2*(q0*q0 + q2*q2) - 1
        r23 = 2*(q2*q3 - q0*q1)
        r31 = 2*(q1*q3 - q0*q2)
        r32 = 2*(q2*q3 + q0*q1)
        r33 = 2*(q0*q0 + q3*q3) - 1

        # 计算Roll、Pitch和Yaw角度（单位为弧度）
        roll = math.atan2(r32, r33)
        pitch = math.atan2(-r31, math.sqrt(r32**2 + r33**2))
        yaw = math.atan2(r21, r11)

        # 将角度转换为度数
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)

        return roll, pitch, yaw

    def euler_to_quaternion(self, roll, pitch, yaw):
        # 将角度转换为弧度
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        # 计算旋转矩阵的元素
        c1 = math.cos(roll/2)
        s1 = math.sin(roll/2)
        c2 = math.cos(pitch/2)
        s2 = math.sin(pitch/2)
        c3 = math.cos(yaw/2)
        s3 = math.sin(yaw/2)

        # 计算四元数的元素
        q0 = c1*c2*c3 + s1*s2*s3
        q1 = s1*c2*c3 - c1*s2*s3
        q2 = c1*s2*c3 + s1*c2*s3
        q3 = c1*c2*s3 - s1*s2*c3

        return [q0, q1, q2, q3]

    