import numpy as np
from collections import namedtuple
from functools import partial
from datetime import timedelta, datetime

import mediapipe_utils as mpu
import depthai as dai
import cv2
from pathlib import Path
from FPS import FPS, now
import time
import sys
from string import Template
import marshal
from robothub_sdk import App, CameraResolution, StreamType, Config, router, Request, PUBLIC_DIR, Stream
from robothub_sdk.device import Device
from rclpy.node import Node, Publisher

# Message imports
from sensor_msgs.msg import Image
from depthai_ros_msgs.msg import HandLandmarkArray, HandLandmark
from std_msgs.msg import Header, ColorRGBA, String
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
import builtin_interfaces.msg

SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
LANDMARK_MODEL_SPARSE = str(SCRIPT_DIR / "models/hand_landmark_sparse_sh4.blob")
DETECTION_POSTPROCESSING_MODEL = str(SCRIPT_DIR / "custom_models/PDPostProcessing_top2_sh1.blob")
MOVENET_LIGHTNING_MODEL = str(SCRIPT_DIR / "models/movenet_singlepose_lightning_U8_transpose.blob")
MOVENET_THUNDER_MODEL = str(SCRIPT_DIR / "models/movenet_singlepose_thunder_U8_transpose.blob")
TEMPLATE_MANAGER_SCRIPT_SOLO = str(SCRIPT_DIR / "template_manager_script_bpf_solo.py")
TEMPLATE_MANAGER_SCRIPT_DUO = str(SCRIPT_DIR / "template_manager_script_bpf_duo.py")

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()



class HandTrackerBpf:
    """
    Mediapipe Hand Tracker for depthai
    Arguments:
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host (Edge mode only),
                    - a file path of an image or a video,
                    - an integer (eg 0) for a webcam id,
                    In edge mode, only "rgb" and "rgb_laconic" are possible
    - pd_model: palm detection model blob file,
    - pd_score: confidence score to determine whether a detection is reliable (a float between 0 and 1).
    - pd_nms_thresh: NMS threshold.
    - use_lm: boolean. When True, run landmark model. Otherwise, only palm detection model is run
    - lm_model: landmark model. Either:
                    - 'full' for LANDMARK_MODEL_FULL,
                    - 'lite' for LANDMARK_MODEL_LITE,
                    - 'sparse' for LANDMARK_MODEL_SPARSE,
                    - a path of a blob file.
    - lm_score_thresh : confidence score to determine whether landmarks prediction is reliable (a float between 0 and 1).
    - use_world_landmarks: boolean. The landmarks model yields 2 types of 3D coordinates : 
                    - coordinates expressed in pixels in the image, always stored in hand.landmarks,
                    - coordinates expressed in meters in the world, stored in hand.world_landmarks 
                    only if use_world_landmarks is True.
    - pp_model: path to the detection post processing model,
    - solo: boolean, when True detect one hand max (much faster since we run the pose detection model only if no hand was detected in the previous frame)
                    On edge mode, always True
    - xyz : boolean, when True calculate the (x, y, z) coords of the detected palms.
    - crop : boolean which indicates if square cropping on source images is applied or not
    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - resolution : sensor resolution "full" (1920x1080) or "ultra" (3840x2160),
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
                    The width is calculated accordingly to height and depends on value of 'crop'
    - use_gesture : boolean, when True, recognize hand poses froma predefined set of poses
                    (ONE, TWO, THREE, FOUR, FIVE, OK, PEACE, FIST)
    - body_pre_focusing: "right" or "left" or "group" or "higher". Body pre focusing is the use
                    of a body pose detector to help to focus on the region of the image that
                    contains one hand ("left" or "right") or "both" hands. 
                    If not in solo mode, body_pre_focusing is forced to 'group'
    - body_model : Movenet single pose model: "lightning", "thunder"
    - body_score_thresh : Movenet score thresh
    - hands_up_only: boolean. When using body_pre_focusing, if hands_up_only is True, consider only hands for which the wrist keypoint
                    is above the elbow keypoint.
    - single_hand_tolerance_thresh (Duo mode only) : In Duo mode, if there is only one hand in a frame, 
                    in order to know when a second hand will appear you need to run the palm detection 
                    in the following frames. Because palm detection is slow, you may want to delay 
                    the next time you will run it. 'single_hand_tolerance_thresh' is the number of 
                    frames during only one hand is detected before palm detection is run again.
    - lm_nb_threads : 1 or 2 (default=2), number of inference threads for the landmark model
    - use_same_image (Edge Duo mode only) : boolean, when True, use the same image when inferring the landmarks of the 2 hands
                    (setReusePreviousImage(True) in the ImageManip node before the landmark model). 
                    When True, the FPS is significantly higher but the skeleton may appear shifted on one of the 2 hands. 
    - stats : boolean, when True, display some statistics when exiting.   
    - trace : int, 0 = no trace, otherwise print some debug messages or show output of ImageManip nodes
            if trace & 1, print application level info like number of palm detections,
            if trace & 2, print lower level info like when a message is sent or received by the manager script node,
            if trace & 4, show in cv2 windows outputs of ImageManip node,
            if trace & 8, save in file tmp_code.py the python code of the manager script node
            Ex: if trace==3, both application and low level info are displayed.   
    """
    def __init__(self,
                pd_model=PALM_DETECTION_MODEL, 
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                use_lm=True,
                lm_model="lite",
                lm_score_thresh=0.5,
                use_world_landmarks=False,
                pp_model = DETECTION_POSTPROCESSING_MODEL,
                solo=True,
                crop=False,
                use_gesture=False,
                body_pre_focusing = 'higher',
                body_model = "thunder",
                body_score_thresh=0.2,
                hands_up_only=True,
                single_hand_tolerance_thresh=10,
                use_same_image=True,
                lm_nb_threads=2,
                stats=False,
                node = None,
                trace=0
                ):
        if node == None:
            raise ValueError("Exiting due to Missing ROS2 node argument. Create a rclpy.create_node() and pass it.")
        self.node = node
        self.bridge = CvBridge()
        self.use_lm = use_lm
        if not use_lm:
            print("use_lm=False is not supported in Edge mode.")
            sys.exit()
        self.pd_model = pd_model
        print(f"Palm detection blob     : {self.pd_model}")
        if lm_model == "full":
            self.lm_model = LANDMARK_MODEL_FULL
        elif lm_model == "lite":
            self.lm_model = LANDMARK_MODEL_LITE
        elif lm_model == "sparse":
                self.lm_model = LANDMARK_MODEL_SPARSE
        else:
            self.lm_model = lm_model
        print(f"Landmark blob           : {self.lm_model}")
        self.pp_model = pp_model
        print(f"PD post processing blob : {self.pp_model}")
        self.solo = solo
        
        self.body_score_thresh = body_score_thresh
        self.body_input_length = 256
        self.hands_up_only = hands_up_only
        if body_model == "lightning":
            self.body_model = MOVENET_LIGHTNING_MODEL
            self.body_input_length = 192 
        else:
            self.body_model = MOVENET_THUNDER_MODEL            
        print(f"Body pose blob          : {self.body_model}")
        if self.solo:
            print("In Solo mode, # of landmark model threads is forced to 1")
            self.lm_nb_threads = 1
            self.body_pre_focusing = body_pre_focusing 
        else:
            assert lm_nb_threads in [1, 2]
            self.lm_nb_threads = lm_nb_threads
            print("In Duo mode, body_pre_focusing is forced to 'group'")
            self.body_pre_focusing = "group"

        self.pd_score_thresh = pd_score_thresh
        # self.pd_nms_thresh = pd_nms_thresh # pd_nms_thresh is hard coded in pp_model
        self.lm_score_thresh = lm_score_thresh

        self.xyz = False
        self.crop = crop 
        self.use_world_landmarks = use_world_landmarks

        self.stats = stats
        self.trace = trace
        self.use_gesture = use_gesture
        self.single_hand_tolerance_thresh = single_hand_tolerance_thresh
        self.use_same_image = use_same_image
        self.image_publisher = self.node.create_publisher(Image, 'depthai/image', 10)
        self.hands_publisher = self.node.create_publisher(HandLandmarkArray, 'depthai/handss_tracklets', 10)
        # self.overlayPublisher = self.node.create_publisher(Image, 'depthai/overlay_image', 10)

    def create_timestamp(self, ts: timedelta) -> builtin_interfaces.msg.Time:
        ts = ts + self._clock_offset
        sec = ts.total_seconds()
        ros_ts = builtin_interfaces.msg.Time(
            sec=int(sec),
            nanosec=int((sec - int(sec)) * 1_000_000_000),
        )
        return ros_ts
 
    def publish_frame(self, frame_id, publisher: Publisher, frame: dai.ImgFrame) -> None:
        # timestamp = self.create_timestamp(frame.getTimestamp())
        timestamp = self.node.get_clock().now().to_msg()
        image_msg = self.bridge.cv2_to_imgmsg(frame.getCvFrame())
        image_msg.header = Header(stamp=timestamp, frame_id=frame_id)
        publisher.publish(image_msg)

    def publish_hands(self, frame_id, publisher, dai_hand) -> None:
        # timestamp = self.create_timestamp(dai_hand.getTimestamp())
        timestamp = self.node.get_clock().now().to_msg()
        res = marshal.loads(dai_hand.getData())
        handMsgs = HandLandmarkArray()

        for i in range(len(res.get("lm_score",[]))):
            hand = self.extract_hand_data(res, i)
            local_msg = HandLandmark()
            local_msg.label = hand.label
            local_msg.lm_score = hand.lm_score
            if hand.gesture != None:
                local_msg.gesture = hand.gesture
            else:
                local_msg.gesture = ''
    
            for x, y in hand.landmarks:
                loc = Pose2D()
                loc.x = float(x)
                loc.y = float(y)
                local_msg.landmark.append(loc)
                x, y, z = hand.xyz
                local_msg.is_spatial = True
                local_msg.position.x = x / 1000
                local_msg.position.y = y / 1000
                local_msg.position.z = z / 1000
            handMsgs.landmarks.append(local_msg)
            if hand.gesture == 'FIST':
                fistFound = True
        
        handMsgs.header.frame_id = frame_id
        handMsgs.header.stamp = timestamp
        publisher.publish(handMsgs)

    def setup(self, device, resolution, internal_fps, xyz, internal_frame_height):
        self.device = device
        self._clock_offset = timedelta(microseconds=(time.time() - time.monotonic()) * 1_000_000)

        # Note that here (in Host mode), specifying "rgb_laconic" has no effect
        # Color camera frames are systematically transferred to the host
        self.input_type = "rgb" # OAK* internal color camera
        # self.laconic = False # Camera frames are not sent to the host
        if resolution == "full":
            self.resolution = (1920, 1080)
        elif resolution == "ultra":
            self.resolution = (3840, 2160)
        else:
            print(f"Error: {resolution} is not a valid resolution !")
            sys.exit()
        print("Sensor resolution:", self.resolution)

        if xyz:
            # Check if the device supports stereo
            cameras = self.device.cameras
            if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras:
                self.xyz = True
            else:
                print("Warning: depth unavailable on this device, 'xyz' argument is ignored")

        if internal_fps is None:
            if self.lm_model == LANDMARK_MODEL_FULL:
                if self.xyz:
                    self.internal_fps = 22 
                else:
                    self.internal_fps = 26 
            elif self.lm_model == LANDMARK_MODEL_LITE:
                if self.xyz:
                    self.internal_fps = 29 
                else:
                    self.internal_fps = 36 
            elif self.lm_model == LANDMARK_MODEL_SPARSE:
                if self.xyz:
                    self.internal_fps = 24 
                else:
                    self.internal_fps = 29 
            else:
                self.internal_fps = 39
        else:
            self.internal_fps = internal_fps 
        print(f"Internal camera FPS set to: {self.internal_fps}") 

        self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps

        if self.crop:
            self.frame_size, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height, self.resolution)
            self.img_h = self.img_w = self.frame_size
            self.pad_w = self.pad_h = 0
            self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2
        else:
            width, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height * self.resolution[0] / self.resolution[1], self.resolution, is_height=False)
            self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
            self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
            self.pad_h = (self.img_w - self.img_h) // 2
            self.pad_w = 0
            self.frame_size = self.img_w
            self.crop_w = 0
    
        print(f"Internal camera image size: {self.img_w} x {self.img_h} - pad_h: {self.pad_h}")

        
        # Defines the default crop region (pads the full image from both sides to make it a square image) 
        # Used when the algorithm cannot reliably determine the crop region from the previous frame.
        self.crop_region = mpu.CropRegion(-self.pad_w, -self.pad_h,-self.pad_w+self.frame_size, -self.pad_h+self.frame_size, self.frame_size)

        # Define and start pipeline
        usb_speed = self.device.usb_speed
        self.create_pipeline()
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        self.fps = FPS()

        self.nb_frames_body_inference = 0
        self.nb_frames_pd_inference = 0
        self.nb_frames_lm_inference = 0
        self.nb_lm_inferences = 0
        self.nb_failed_lm_inferences = 0
        self.nb_frames_lm_inference_after_landmarks_ROI = 0
        self.nb_frames_no_hand = 0


    def setupQueue(self):
                # Define data queues 
        # if not self.laconic:
        #     self.q_video = self.device.internal.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        # self.q_manager_out = self.device.internal.getOutputQueue(name="manager_out", maxSize=1, blocking=False)
        # # For showing outputs of ImageManip nodes (debugging)
        # if self.trace & 4:
        #     self.q_pre_body_manip_out = self.device.internal.getOutputQueue(name="pre_body_manip_out", maxSize=1, blocking=False)
        #     self.q_pre_pd_manip_out = self.device.internal.getOutputQueue(name="pre_pd_manip_out", maxSize=1, blocking=False)
        #     self.q_pre_lm_manip_out = self.device.internal.getOutputQueue(name="pre_lm_manip_out", maxSize=1, blocking=False)    

        self.fps = FPS()

        self.nb_frames_body_inference = 0
        self.nb_frames_pd_inference = 0
        self.nb_frames_lm_inference = 0
        self.nb_lm_inferences = 0
        self.nb_failed_lm_inferences = 0
        self.nb_frames_lm_inference_after_landmarks_ROI = 0
        self.nb_frames_no_hand = 0


    def create_pipeline(self):
        print("Creating pipeline...")
        # pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)
        self.device.pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)
        self.pd_input_length = 128

        # ColorCamera
        print("Creating Color Camera...")
        # cam = pipeline.createColorCamera()

 
        if self.resolution[0] == 1920:
            sensorRes = CameraResolution.THE_1080_P
        else:
            sensorRes = CameraResolution.THE_4_K
        # cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        # cam.setInterleaved(False)
        # cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        # cam.setFps(self.internal_fps)

        
        if self.crop:
            width = self.frame_size
            height = self.frame_size
        else:
            width = self.img_w 
            height = self.img_h
            # cam.setVideoSize(self.img_w, self.img_h)
            # cam.setPreviewSize(self.img_w, self.img_h)

        cam = self.device.configure_camera(dai.CameraBoardSocket.RGB, res=sensorRes, fps=self.internal_fps, preview_size=(width, height), video_size=(width, height), isp_scale = (self.scale_nd[0], self.scale_nd[1])) 

        # TODO(Sachin): Come back and strucutre the node script with details from below 
        # manager_script = pipeline.create(dai.node.Script)
        # manager_script.setScript(self.build_manager_script(),)
        script_code = self.build_manager_script()
        print(type(script_code))
        manager_script = self.device.create_script(script_code, name='hand_manager')

        if self.xyz:
            print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes...")
            # For now, RGB needs fixed focus to properly align with depth.
            # The value used during calibration should be used here
            calib_data = self.device.internal.readCalibration()
            calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.RGB)
            print(f"Can't set lens position on RoboHub yet: {calib_lens_pos}")
            cam.initialControl.setManualFocus(calib_lens_pos)
            
            mono_resolution = CameraResolution.THE_400_P
            self.device.configure_camera(dai.CameraBoardSocket.LEFT,
                                        res=mono_resolution,
                                        fps=self.internal_fps)

            self.device.configure_camera(dai.CameraBoardSocket.RIGHT,
                                        res=mono_resolution,
                                        fps=self.internal_fps)

            stereo = self.device.create_stereo_depth(confidence_threshold = 230)

            # LR-check is required for depth alignment
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            stereo.setSubpixel(False)  # subpixel True brings latency
            # MEDIAN_OFF necessary in depthai 2.7.2. 
            # Otherwise : [critical] Fatal error. Please report to developers. Log: 'StereoSipp' '533'
            # stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)

            spatial_location_calculator = self.device.pipeline.createSpatialLocationCalculator()
            spatial_location_calculator.setWaitForConfigInput(True)
            spatial_location_calculator.inputDepth.setBlocking(False)
            spatial_location_calculator.inputDepth.setQueueSize(1)

            # left.out.link(stereo.left)
            # right.out.link(stereo.right)    

            stereo.depth.link(spatial_location_calculator.inputDepth)

            manager_script.outputs['spatial_location_config'].link(spatial_location_calculator.inputConfig)
            spatial_location_calculator.out.link(manager_script.inputs['spatial_data'])            

        # Define body pose detection pre processing: resize preview to (self.body_input_length, self.body_input_length)
        # and transform BGR to RGB
        print("Creating Body Pose Detection pre processing image manip...")
        (pre_body_manip, pre_body_manip_stream) = self.device.create_image_manipulator()

        # pre_body_manip = self.device.pipeline.create(dai.node.ImageManip)
        pre_body_manip.setMaxOutputFrameSize(self.body_input_length*self.body_input_length*3)
        pre_body_manip.setWaitForConfigInput(True)
        pre_body_manip.inputImage.setQueueSize(1)
        pre_body_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_body_manip.inputImage)
        manager_script.outputs['pre_body_manip_cfg'].link(pre_body_manip.inputConfig)
        # For debugging
        if self.trace & 4:
            # TODO(sachin): Add a callback for this. 
            pre_body_manip_stream.consume()

        # Define landmark model
        # self.pre_body_manip_stream = Stream(pre_body_manip, pre_body_manip.out, stream_type=StreamType.FRAME, description="pre_body_manip_cfg Out" )
        print("Creating Body Pose Detection Neural Network...")
        (body_nn, body_nn_det_out, body_nn_det_passthrough) = self.device.create_nn(pre_body_manip_stream, Path(self.body_model))

        # body_nn = self.device.pipeline.create(dai.node.NeuralNetwork)
        # body_nn.setBlobPath(self.body_model)
        # lm_nn.setNumInferenceThreads(1)
        # pre_body_manip.out.link(body_nn.input)
        body_nn.out.link(manager_script.inputs['from_body_nn'])

        # Define palm detection pre processing: resize preview to (self.pd_input_length, self.pd_input_length)
        print("Creating Palm Detection pre processing image manip...")
        (pre_pd_manip, pre_pd_manip_stream) = self.device.create_image_manipulator()

        # pre_pd_manip = self.device.pipeline.create(dai.node.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_pd_manip.inputImage)
        manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)

        # For debugging
        if self.trace & 4:
            # TODO(sachin): Add a callback for this. 
            pre_pd_manip_stream.consume()
            # pre_pd_manip_out = self.device.pipeline.createXLinkOut()
            # pre_pd_manip_out.setStreamName("pre_pd_manip_out")
            # pre_pd_manip.out.link(pre_pd_manip_out.input)

        # Define palm detection model
        print("Creating Palm Detection Neural Network...")
        # self.pre_pd_manip_stream = Stream(pre_pd_manip, pre_pd_manip.out, stream_type=StreamType.FRAME, description="pre_pd_manip_cfg Out" )
        (pd_nn, pd_nn_out, pd_nn_passthrough) = self.device.create_nn(pre_pd_manip_stream, Path(self.pd_model))


        # pd_nn = self.device.pipeline.create(dai.node.NeuralNetwork)
        # pd_nn.setBlobPath(self.pd_model)
        # pre_pd_manip.out.link(pd_nn.input)

        # Define pose detection post processing "model"
        print("Creating Palm Detection post processing Neural Network...")
        (post_pd_nn, post_pd_nn_out, post_pd_nn_passthrough) = self.device.create_nn(pd_nn_out, Path(self.pp_model))

        # post_pd_nn = self.device.pipeline.create(dai.node.NeuralNetwork)
        # post_pd_nn.setBlobPath(self.pp_model)
        # pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs['from_post_pd_nn'])
        
        # Define link to send result to host 
        # manager_out = self.device.pipeline.create(dai.node.XLinkOut)
        # manager_out.setStreamName("manager_out")
        # manager_script.outputs['host'].link(manager_out.input)

        # self.cropped_script = self.device.create_script(script_path=Path("./script.py"),
        #     inputs={
        #         'frames': self.device.streams.color_video
        #     },
        #     outputs={
        #         'manip_cfg': self.manip.inputConfig,
        #         'manip_img': self.manip.inputImage,
        #     })

        # Define landmark pre processing image manip
        print("Creating Hand Landmark pre processing image manip...") 
        self.lm_input_length = 224
        (pre_lm_manip, pre_lm_manip_stream) = self.device.create_image_manipulator()
        pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length*self.lm_input_length*3)
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_lm_manip.inputImage)

        # For debugging
        if self.trace & 4:
        # TODO(sachin): Add a callback for this. 
            pre_lm_manip_stream.consume()

        manager_script.outputs['pre_lm_manip_cfg'].link(pre_lm_manip.inputConfig)

        # Define landmark model
        print(f"Creating Hand Landmark Neural Network ({'1 thread' if self.lm_nb_threads == 1 else '2 threads'})...")   
        (lm_nn, lm_nn_out, lm_nn_passthrough) = self.device.create_nn(pre_lm_manip_stream, Path(self.lm_model))
       
        # lm_nn = self.device.pipeline.create(dai.node.NeuralNetwork)
        # lm_nn.setBlobPath(self.lm_model)
        lm_nn.setNumInferenceThreads(self.lm_nb_threads)
        # pre_lm_manip.out.link(lm_nn.input)
        lm_nn.out.link(manager_script.inputs['from_lm_nn'])
        
        # self.encoder = self.device.create_encoder(
        #                 self.device.streams.color_still.output_node,
        #                 fps=8,
        #                 profile=dai.VideoEncoderProperties.Profile.MJPEG,
        #                 quality=80,
        #             )
        self.device.streams.create(
                manager_script,
                manager_script.outputs['host'],
                stream_type=StreamType.BINARY,
                rate=27,
                description="manager_script_host_out"
            ).consume(partial(self.publish_hands, "color_ccm_frame", self.hands_publisher))
        # if not self.laconic:
            # TODO(sachin): add connections here
            # cam_out = self.device.pipeline.createXLinkOut()
            # cam_out.setStreamName("cam_out")
            # cam_out.input.setQueueSize(1)
            # cam_out.input.setBlocking(False)
            # cam.video.link(cam_out.input)
        self.device.streams.color_video.consume(partial(self.publish_frame, "color_ccm_frame", self.image_publisher))

        print("Pipeline created.")
        # return pipeline        
    
    def build_manager_script(self):
        '''
        The code of the scripting node 'manager_script' depends on :
            - the score threshold,
            - the video frame shape
        So we build this code from the content of the file template_manager_script_*.py which is a python template
        '''
        # Read the template
        with open(TEMPLATE_MANAGER_SCRIPT_SOLO if self.solo else TEMPLATE_MANAGER_SCRIPT_DUO, 'r') as file:
            template = Template(file.read())

        # Perform the substitution
        code = template.substitute(
                    _TRACE1 = "node.warn" if self.trace & 1 else "#",
                    _TRACE2 = "node.warn" if self.trace & 2 else "#",
                    _pd_score_thresh = self.pd_score_thresh,
                    _lm_score_thresh = self.lm_score_thresh,
                    _pad_h = self.pad_h,
                    _img_h = self.img_h,
                    _img_w = self.img_w,
                    _frame_size = self.frame_size,
                    _crop_w = self.crop_w,
                    _IF_XYZ = "" if self.xyz else '"""',
                    _body_pre_focusing = self.body_pre_focusing,
                    _body_score_thresh = self.body_score_thresh,
                    _body_input_length = self.body_input_length,
                    _hands_up_only = self.hands_up_only,
                    _single_hand_tolerance_thresh= self.single_hand_tolerance_thresh,
                    _IF_USE_SAME_IMAGE = "" if self.use_same_image else '"""',
                    _IF_USE_WORLD_LANDMARKS = "" if self.use_world_landmarks else '"""',
        )
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)
        # For debugging
        if self.trace & 8:
            with open("tmp_code.py", "w") as file:
                file.write(code)

        return code

    def extract_hand_data(self, res, hand_idx):
        hand = mpu.HandRegion()
        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * self.frame_size
        hand.rotation = res["rotation"][hand_idx] 
        hand.rect_points = mpu.rotated_rect_to_points(hand.rect_x_center_a, hand.rect_y_center_a, hand.rect_w_a, hand.rect_h_a, hand.rotation)
        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res['rrn_lms'][hand_idx]).reshape(-1,3)
        hand.landmarks = (np.array(res["sqn_lms"][hand_idx]) * self.frame_size).reshape(-1,2).astype(np.int)
        if self.xyz:
            hand.xyz = np.array(res["xyz"][hand_idx])
            hand.xyz_zone = res["xyz_zone"][hand_idx]
        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
        if self.pad_h > 0:
            hand.landmarks[:,1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            hand.landmarks[:,0] -= self.pad_w
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][0] -= self.pad_w

        # World landmarks
        if self.use_world_landmarks:
            hand.world_landmarks = np.array(res["world_lms"][hand_idx]).reshape(-1, 3)

        if self.use_gesture: mpu.recognize_gesture(hand)

        return hand

    def next_frame(self):

        self.fps.update()

        if self.laconic:
            video_frame = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            # in_video = self.q_video.get()
            video_frame = self.device.streams.color_video.last_value.getCvFrame() if self.device.streams.color_video.last_value is not None else None      

        # For debugging
        if self.trace & 4:
            pre_body_manip = self.q_pre_body_manip_out.tryGet()
            if pre_body_manip:
                pre_pd_manip = pre_body_manip.getCvFrame()
                cv2.imshow("pre_body_manip", pre_pd_manip)
            pre_pd_manip = self.q_pre_pd_manip_out.tryGet()
            if pre_pd_manip:
                pre_pd_manip = pre_pd_manip.getCvFrame()
                cv2.imshow("pre_pd_manip", pre_pd_manip)
            pre_lm_manip = self.q_pre_lm_manip_out.tryGet()
            if pre_lm_manip:
                pre_lm_manip = pre_lm_manip.getCvFrame()
                cv2.imshow("pre_lm_manip", pre_lm_manip)

        # Get result from device
        res = marshal.loads(self.q_manager_out.get().getData())
        hands = []
        for i in range(len(res.get("lm_score",[]))):
            hand = self.extract_hand_data(res, i)
            hands.append(hand)


        # Statistics
        if self.stats:
            if res["bd_pd_inf"] == 1:
                self.nb_frames_body_inference += 1
            elif res["bd_pd_inf"] == 2:
                self.nb_frames_body_inference += 1
                self.nb_frames_pd_inference += 1
            else:
                if res["nb_lm_inf"] > 0:
                     self.nb_frames_lm_inference_after_landmarks_ROI += 1
            if res["nb_lm_inf"] == 0:
                self.nb_frames_no_hand += 1
            else:
                self.nb_frames_lm_inference += 1
                self.nb_lm_inferences += res["nb_lm_inf"]
                self.nb_failed_lm_inferences += res["nb_lm_inf"] - len(hands)

        return video_frame, hands, None


    def exit(self):
        self.device.close()
        # Print some stats
        if self.stats:
            nb_frames = self.fps.nb_frames()
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {nb_frames})")
            print(f"# frames w/ no hand           : {self.nb_frames_no_hand} ({100*self.nb_frames_no_hand/nb_frames:.1f}%)")
            print(f"# frames w/ body detection    : {self.nb_frames_body_inference} ({100*self.nb_frames_body_inference/nb_frames:.1f}%)")
            print(f"# frames w/ palm detection    : {self.nb_frames_pd_inference} ({100*self.nb_frames_pd_inference/nb_frames:.1f}%)")
            print(f"# frames w/ landmark inference : {self.nb_frames_lm_inference} ({100*self.nb_frames_lm_inference/nb_frames:.1f}%)- # after palm detection: {self.nb_frames_lm_inference - self.nb_frames_lm_inference_after_landmarks_ROI} - # after landmarks ROI prediction: {self.nb_frames_lm_inference_after_landmarks_ROI}")
            if not self.solo:
                print(f"On frames with at least one landmark inference, average number of landmarks inferences/frame: {self.nb_lm_inferences/self.nb_frames_lm_inference:.2f}")
            if self.nb_lm_inferences:
                print(f"# lm inferences: {self.nb_lm_inferences} - # failed lm inferences: {self.nb_failed_lm_inferences} ({100*self.nb_failed_lm_inferences/self.nb_lm_inferences:.1f}%)")