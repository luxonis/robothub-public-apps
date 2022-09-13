# from HandTrackerRenderer import HandTrackerRenderer
from HandTrackerBpfEdge import HandTrackerBpf
from HandTrackerRenderer import HandTrackerRenderer

from functools import partial
from typing import List
import time
import datetime

#Depthai Imports
import depthai as dai
import numpy as np

# ROS2 Imports
import rclpy
from rclpy.node import Node

# Message imports
from sensor_msgs.msg import Image
from depthai_ros_msgs.msg import HandLandmarkArray, HandLandmark
from std_msgs.msg import String
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge

from robothub_sdk import (
    App,
    CameraResolution,
    InputStream,
    StreamType,
    Config,
)

class HandTracker(App):
    def on_initialize(self, devices: List[dai.DeviceInfo]):
        rclpy.init()
        self.node = rclpy.create_node('HandTracker')
        self.bridge = CvBridge()
        self.camera_controls = []
        
    def on_exit(self):
        self.node.destroy_node()
        rclpy.shutdown() 

    def on_configuration(self, old_configuration: Config):
        print("Configuration update", self.config.values())
        self.imagePublisher = self.node.create_publisher(Image, 'depthai/image', 10)
        self.overlayPublisher = self.node.create_publisher(Image, 'depthai/overlay_image', 10)
        self.handsPublisher = self.node.create_publisher(HandLandmarkArray, 'depthai/handss_tracklets', 10)
        
        self.tracker = HandTrackerBpf(
            use_lm= self.config.use_lm,
            use_world_landmarks=self.config.use_world_landmarks,
            use_gesture=self.config.use_gesture,
            xyz=True,
            solo=False,
            crop=False,
            body_pre_focusing='group',
            hands_up_only= not self.config.all_hands,
            single_hand_tolerance_thresh=self.config.single_hand_tolerance_thresh,
            lm_nb_threads=self.config.lm_nb_threads,
            stats=True,
            trace=self.config.trace,
            )

    def on_setup(self, device: Device):
        self.tracker.setup(device, self.config.resolution, self.config.internal_fps, self.config.xyz, self.config.internal_frame_height)
        self.camera_controls.append(device.streams.color_control) # adding Capture control
        self.renderer = HandTrackerRenderer(tracker=self.tracker)
    
    def on_update(self):
        frame, hands, bag = self.tracker.next_frame()
        frame_vis = self.renderer.draw(frame, hands, bag)
        handMsgs = HandLandmarkArray()
        fistFound = False

        for hand in hands:
            local_msg = HandLandmark()
            local_msg.label = hand.label
            local_msg.lm_score = hand.lm_score
            if hand.gesture != None:
                local_msg.gesture = hand.gesture
            else:
                local_msg.gesture = ''
    
            for x, y in hand.landmarks:
                loc = Pose2D()
                loc.x = x
                loc.y = y
                local_msg.landmark.append(loc)
                x, y, z = hand.xyz
                local_msg.is_spatial = True
                local_msg.position.x = x
                local_msg.position.y = y
                local_msg.position.z = z
            handMsgs.landmarks.append(local_msg)

            if hand.gesture == 'FIST':
                fistFound = True

        handMsgs.header.frame_id = ""
        handMsgs.header.stamp = self.node.get_clock().now().to_msg()

        self.imagePublisher.publish(self.bridge.cv2_to_imgmsg(frame))
        self.overlayPublisher.publish(self.bridge.cv2_to_imgmsg(frame_vis))
        self.handsPublisher.publish(handMsgs)

        rclpy.spin_once(self.node)
        if fistFound:
            time.sleep(5)
            for camera_control in self.camera_controls:
                ctl = dai.CameraControl()
                ctl.setCaptureStill(True)
                camera_control.send(ctl)


        
 