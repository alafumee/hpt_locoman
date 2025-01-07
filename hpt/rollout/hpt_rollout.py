import argparse
import os
import time

import hydra
from omegaconf import OmegaConf
import rospy
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np
from config.config import Cfg
from utilities.orientation_utils_numpy import rot_mat_to_rpy, rpy_to_rot_mat
from teleoperation.tele_vision import OpenTeleVision
from sensor_msgs.msg import Image
import ros_numpy
import cv2
import threading
from teleoperation.camera_utils import list_video_devices, find_device_path_by_name
from multiprocessing import Array, Process, shared_memory
import sys
import signal
import h5py
from datetime import datetime
from pynput import keyboard
import torch
from einops import rearrange
import pickle
import yaml
from scipy.spatial.transform import Rotation as R

import hpt.utils.utils as utils

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Rollout():
    def __init__(self, hydra_cfg, config=None, policy_config=None):
        rospy.init_node('rollout')
        self.hydra_conf = hydra_cfg
        self.command_publisher = rospy.Publisher(Cfg.commander.auto_policy_topic, Float32MultiArray, queue_size=1)
        self.fsm_publisher = rospy.Publisher(Cfg.fsm_switcher.fsm_state_topic, Int32, queue_size = 1)
        self.fsm_to_teleop_mode_mapping = Cfg.teleoperation.human_teleoperator.fsm_to_teleop_mode_mapping
        self.model_time = 0
        
        self.device = hydra_cfg.device
        self.delta_act_dict = {}
        self.act_dict = {}
        
        if self.hydra_conf.use_real_robot:
            self.init_cameras()
        else:
            # simulation view subscriber
            self.sim_view_subscriber = rospy.Subscriber(Cfg.teleoperation.teleop_view_topic, Image, self.sim_view_callback)
            img_shape = (Cfg.teleoperation.fpv_height, Cfg.teleoperation.fpv_width, 3)
            self.sim_view_frame = np.zeros(img_shape, dtype=np.uint8)
        # command buffer
        self.command = np.zeros(20)  # body: xyzrpy, eef_r: xyzrpy, eef_l: xyzrpy, grippers: 2 angles
        self.command_msg = Float32MultiArray()
        self.command_msg.data = self.command.tolist()
        self.robot_command = np.zeros(20)
        
        self.fsm_state_msg = Int32()
        self.fsm_state_msg.data = 0

        # initial values
        self.initial_receive = True
        self.init_body_pos = np.zeros(3)
        self.init_body_rot = np.eye(3)
        self.init_eef_pos = np.zeros((2, 3))
        self.init_eef_rot = np.array([np.eye(3), np.eye(3)])
        self.init_gripper_angles = np.zeros(2)

        self.is_changing_reveive_status = False
        self.begin_to_receive = False
        self.reset_signal_subscriber = rospy.Subscriber(Cfg.teleoperation.receive_action_topic, Int32, self.reset_signal_callback, queue_size=1)
        self.teleoperation_mode = Cfg.teleoperation.human_teleoperator.mode
        self.is_changing_teleop_mode = False
        self.teleop_mode_subscriber = rospy.Subscriber(Cfg.teleoperation.human_teleoperator.mode_updata_topic, Int32, self.teleop_mode_callback, queue_size=1)
        self.rate = rospy.Rate(200)
        
        self.robot_reset_publisher = rospy.Publisher(Cfg.teleoperation.robot_reset_topic, Int32, queue_size = 1)
        self.robot_reset_msg = Int32()
        self.robot_reset_msg.data = 0

        self.robot_reset_pose_subscriber = rospy.Subscriber(Cfg.commander.robot_reset_pose_topic, Float32MultiArray, self.robot_reset_pose_callback, queue_size=1)
        self.robot_reset_pose = np.zeros(20)
        self.on_reset = False
        self.reset_finished = False

        self.robot_state_subscriber = rospy.Subscriber(Cfg.teleoperation.robot_state_topic, Float32MultiArray, self.robot_proprio_state_callback, queue_size=1)
        
        # self.pinch_gripper_angle_scale = 10.0
        self.pinch_dist_gripper_full_close = 0.02
        self.pinch_dist_gripper_full_open = 0.15
        self.grippr_full_close_angle = Cfg.commander.gripper_angle_range[0]
        self.grippr_full_open_angle = Cfg.commander.gripper_angle_range[1]
        self.eef_xyz_scale = 1.0
        self.manipulate_eef_idx = 0

        self.init_embodiment_proprio_states()
        self.get_embodiment_masks(embodiment='locoman')
        self.bc_policy = self.init_policy()
        # actions masks
        self.act_mask_dict = {}
        self.act_mask_dict['delta_body_pose'] = torch.tensor(self.act_body_mask).to(self.device).unsqueeze(0)
        self.act_mask_dict['delta_eef_pose'] = torch.tensor(self.act_eef_mask).to(self.device).unsqueeze(0)
        self.act_mask_dict['delta_gripper'] = torch.tensor(self.act_gripper_mask).to(self.device).unsqueeze(0)
        
        self.pause_commands = False

        self.head_cam_image_history = []
        self.wrist_cam_image_history = []
        self.command_history = []
        self.robot_state_history = []
        
        self.rollout_counter = 0
        self.action_idx = 0
        action_horizon = 60
        self.command_traj = np.zeros((action_horizon, 20))
        self.command_trajs = []
        self.infer_interval = self.hydra_conf.inference_interval
        self.chunk_size = self.hydra_conf.action_chunk_size
        
        # prepare for data processing and collection
        self.body_xyz_scale = Cfg.teleoperation.human_teleoperator.body_xyz_scale
        self.body_rpy_scale = Cfg.teleoperation.human_teleoperator.body_rpy_scale
        self.eef_xyz_scale = Cfg.teleoperation.human_teleoperator.eef_xyz_scale
        self.eef_rpy_scale = Cfg.teleoperation.human_teleoperator.eef_rpy_scale     
        self.gripper_angle_scale = Cfg.teleoperation.human_teleoperator.gripper_angle_scale
        self.human_command_body_rpy_range = np.array([Cfg.teleoperation.human_teleoperator.body_r_range,
                                                      Cfg.teleoperation.human_teleoperator.body_p_range,
                                                      Cfg.teleoperation.human_teleoperator.body_y_range,])
        
        self.state_flag = 0
    
    def reset_signal_callback(self, msg):
        self.is_changing_reveive_status = True
        if msg.data == 0:
            self.begin_to_receive = False
            self.initial_receive = True
            print("No longer receiving. Initial receive status reset.")
        elif msg.data == 1:
            self.begin_to_receive = True
            self.initial_receive = True
            self.fsm_state_msg.data = 3 # 3 for right gripper, 1 for right foot
            self.teleoperation_mode = self.fsm_to_teleop_mode_mapping[self.fsm_state_msg.data]
            self.fsm_publisher.publish(self.fsm_state_msg)
            print("Begin to receive. Initial receive status reset.")
        elif msg.data == 2:
            self.begin_to_receive = True
            self.initial_receive = True
            print("Ready to record!")
        self.is_changing_reveive_status = False
    
    def update_manipulate_eef_idx(self):
        if self.teleoperation_mode == 1:
            # right gripper manipulation
            self.manipulate_eef_idx = 0
        elif self.teleoperation_mode == 2:
            # left gripper manipulation
            self.manipulate_eef_idx = 1
        
    def teleop_mode_callback(self, msg):
        self.is_changing_teleop_mode = True
        self.teleoperation_mode = msg.data
        print(f"Teleoperation mode updated to {self.teleoperation_mode}.")
        self.is_changing_teleop_mode = False

    def sim_view_callback(self, msg):
        self.sim_view_frame = ros_numpy.numpify(msg)
        
    def robot_state_callback(self, msg):
        # update joint positions
        self.robot_state = np.array(msg.data)

    def robot_reset_pose_callback(self, msg):
        self.robot_reset_pose = np.array(msg.data)
        if self.on_reset:
            self.reset_finished = True
            self.on_reset = False
            print('robot reset pose', self.robot_reset_pose)

    def robot_proprio_state_callback(self, msg):
        # update proprio states of locoman
        self.body_pose_prorio_callback = np.array(msg.data)[:6]
        self.eef_pose_proprio_callback = np.array(msg.data)[6:18]
        self.gripper_angle_proprio_callback = np.array(msg.data)[18:20]
        self.joint_pos_proprio_callback = np.array(msg.data)[20:38]
        self.joint_vel_proprio_callback = np.array(msg.data)[38:56]

    def init_embodiment_proprio_states(self):
        self.body_pose_prorio_callback = np.zeros(6)
        self.eef_pose_proprio_callback = np.zeros(12)
        self.eef_to_body_pose_proprio_callback = np.zeros(12)
        self.gripper_angle_proprio_callback = np.zeros(2)
        self.joint_pos_proprio_callback = np.zeros(18)
        self.joint_vel_proprio_callback = np.zeros(18)

        self.body_pose_prorio = np.zeros(6)
        self.eef_pose_proprio = np.zeros(12)
        self.eef_to_body_pose_proprio = np.zeros(12)
        self.gripper_angle_proprio = np.zeros(2)
        self.joint_pos_proprio = np.zeros(18)
        self.joint_vel_proprio = np.zeros(18)

    def init_cameras(self):
        import pyrealsense2 as rs
        # initialize all cameras
        self.desired_stream_fps = self.hydra_conf.desired_stream_fps
        # initialize head camera
        # realsense as head camera
        if self.hydra_conf.head_camera_type == 0:
            self.head_camera_resolution = Cfg.teleoperation.realsense_resolution
            self.head_frame_res = Cfg.teleoperation.head_view_resolution
            self.head_color_frame = np.zeros((self.head_frame_res[0], self.head_frame_res[1], 3), dtype=np.uint8)
            self.head_cam_pipeline = rs.pipeline()
            self.head_cam_config = rs.config()
            pipeline_wrapper = rs.pipeline_wrapper(self.head_cam_pipeline)
            pipeline_profile = self.head_cam_config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("a head camera is required for real-robot teleoperation")
                exit(0)
            self.head_cam_config.enable_stream(rs.stream.color, self.head_camera_resolution[1], self.head_camera_resolution[0], rs.format.bgr8, 30)
            # start streaming head cam
            self.head_cam_pipeline.start(self.head_cam_config)
        # stereo rgb camera (dual lens) as the head camera 
        elif self.hydra_conf.head_camera_type == 1:
            # self.head_camera_resolution: the supported resoltion of the camera, to get the original frame without cropping
            self.head_camera_resolution = Cfg.teleoperation.stereo_rgb_resolution
            # self.head_view_resolution: the resolition of the images that are seen and recorded
            self.head_view_resolution = Cfg.teleoperation.head_view_resolution
            self.crop_size_w = 0
            self.crop_size_h = 0
            self.head_frame_res = (self.head_view_resolution[0] - self.crop_size_h, self.head_view_resolution[1] - 2 * self.crop_size_w)
            self.head_color_frame = np.zeros((self.head_frame_res[0], 2 * self.head_frame_res[1], 3), dtype=np.uint8)
            device_map = list_video_devices()
            head_camera_name = "3D USB Camera"
            device_path = find_device_path_by_name(device_map, head_camera_name)
            self.head_cap = cv2.VideoCapture(device_path)
            self.head_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.head_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * self.head_camera_resolution[1])
            self.head_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.head_camera_resolution[0])
            self.head_cap.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)
        else:
            raise NotImplementedError("Not supported camera.")
        
        # initialize wrist camera/cameras
        if self.hydra_conf.use_wrist_camera:
            self.wrist_camera_resolution = Cfg.teleoperation.rgb_resolution
            self.wrist_view_resolution = Cfg.teleoperation.wrist_view_resolution
            self.wrist_color_frame = np.zeros((self.wrist_view_resolution[0], self.wrist_view_resolution[1], 3), dtype=np.uint8)
            device_map = list_video_devices()
            wrist_camera_name = "Global Shutter Camera"
            device_path = find_device_path_by_name(device_map, wrist_camera_name)
            self.wrist_cap1 = cv2.VideoCapture(device_path)
            self.wrist_cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.wrist_cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.wrist_camera_resolution[1])
            self.wrist_cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.wrist_camera_resolution[0])
            self.wrist_cap1.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)

    def head_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        if self.hydra_conf.head_camera_type == 0:
            try:
                while not rospy.is_shutdown():
                    start_time = time.time()
                    # handle head camera (realsense) streaming 
                    frames = self.head_cam_pipeline.wait_for_frames()
                    # depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        return
                    head_color_frame = np.asanyarray(color_frame.get_data())
                    head_color_frame = cv2.resize(head_color_frame, (self.head_frame_res[1], self.head_frame_res[0]))
                    self.head_color_frame = cv2.cvtColor(head_color_frame, cv2.COLOR_BGR2RGB)
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
            finally:
                self.head_cam_pipeline.stop()
        elif self.hydra_conf.head_camera_type == 1:
            try:
                while not rospy.is_shutdown():
                    start_time = time.time()
                    ret, frame = self.head_cap.read()
                    # print('frame 0', frame.shape)
                    frame = cv2.resize(frame, (2 * self.head_frame_res[1], self.head_frame_res[0]))
                    # print('frame 1', frame.shape)
                    image_left = frame[:, :self.head_frame_res[1], :]
                    # print('image_left', image_left.shape)
                    image_right = frame[:, self.head_frame_res[1]:, :]
                    # print('image right', image_right.shape)
                    if self.crop_size_w != 0:
                        bgr = np.hstack((image_left[self.crop_size_h:, self.crop_size_w:-self.crop_size_w],
                                        image_right[self.crop_size_h:, self.crop_size_w:-self.crop_size_w]))
                    else:
                        bgr = np.hstack((image_left[self.crop_size_h:, :],
                                        image_right[self.crop_size_h:, :]))

                    self.head_color_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    # print('self.head_color_frame', self.head_color_frame.shape)
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    # print(1/(time.time() - start_time))
            finally:
                self.head_cap.release()
        else:
            raise NotImplementedError('Not supported camera.')
        
    def wrist_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        try:
            while not rospy.is_shutdown():
                start_time = time.time()
                ret, frame = self.wrist_cap1.read()
                wrist_color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print('wrist_color_frame', wrist_color_frame.shape)
                self.wrist_color_frame = cv2.resize(wrist_color_frame, (self.wrist_view_resolution[1], self.wrist_view_resolution[0]))
                # print('self.wrist_color_frame', self.wrist_color_frame.shape)
                elapsed_time = time.time() - start_time
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # print(1/(time.time() - start_time))
        finally:
            self.head_cap.release()


    # def init_policy(self):
    #     # dataset_dir = ['/home/yaru/research/locoman_learning/human2locoman/demonstrations/single_toy_collect/locoman']
    #     dataset_dir = ['/home/yaru/research/locoman_learning/human2locoman/demonstrations/toy_collect/locoman']
    #     # dataset_dir = ['/home/yaru/research/locoman_learning/human2locoman/demonstrations/single_toy_collect/locoman/20241126_030550']
    #     policy_class = 'cross'
    #     backbone = 'dinov2'

    #     with open('/home/yaru/research/locoman_learning/human2locoman/HIT/detr/models/cross_transformer/embodiments.yaml', 'r') as f:
    #         emb_dict = yaml.load(f, Loader=yaml.FullLoader)
    #     with open('/home/yaru/research/locoman_learning/human2locoman/HIT/detr/models/cross_transformer/transformer_trunk.yaml', 'r') as f:
    #         transformer_dict = yaml.load(f, Loader=yaml.FullLoader)

    #     def flatten_list(l):
    #         return [item for sublist in l for item in sublist]
    #     name_filter = lambda _: True

    #     dataset_path_list_list = [find_all_hdf5(dataset_dir, False) for dataset_dir in dataset_dir]
    #     num_episodes_0 = len(dataset_path_list_list[0])
    #     dataset_path_list = flatten_list(dataset_path_list_list)
    #     dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    #     dataset_path_list = sorted(dataset_path_list)
    #     stats_dict, _, _ = get_norm_stats(dataset_path_list, emb_dict)

    #     policy_config = {
    #         # 'lr': 1e-5,
    #         # 'lr_backbone': 1e-5,
    #         # 'backbone': backbone,
    #         'norm_stats': stats_dict, 
    #         "embodiment_args_dict": emb_dict,
    #         "transformer_args": transformer_dict,
    #         }
        
    #     config = {
    #         # 'pretrained_path': '/home/yaru/research/locoman_learning/human2locoman/checkpoints/nov27_test_single_toy_collect_bs24_cs60_cross',
    #         # 'pretrained_path': '/home/yaru/research/locoman_learning/human2locoman/checkpoints/not_delta_action_single_toy_collect_bs24_cs60_cross',
    #         'pretrained_path': '/home/yaru/research/locoman_learning/human2locoman/checkpoints/not_delta_action_toy_collect_bs24_cs60_cross',
    #     }

    #     policy = make_policy(policy_class, policy_config)
    #     loading_status = policy.deserialize(torch.load(f'{config["pretrained_path"]}/policy_last.ckpt', map_location='cuda'), eval=True)
    #     print(f'loaded! {loading_status}')
    #     self.bc_model = policy

    def init_policy(self):
        """
        Initialize the policy and load the pretrained model if available.

        Args:
            cfg (Config): The configuration object.
            dataset (Dataset): The dataset object.
            domain (str): The domain of the policy.
            device (str): The device to use for computation.

        Returns:
            Policy: The initialized policy.

        """
        self.hydra_conf.output_dir = self.hydra_conf.output_dir + "/" + str(self.hydra_conf.seed)
        utils.set_seed(self.hydra_conf.seed)
        print(self.hydra_conf)

        device = "cuda"
        domain_list = [d.strip() for d in self.hydra_conf.domains.split(",")]
        domain = domain_list[0]

        # initialize policy
        policy = hydra.utils.instantiate(self.hydra_conf.network).to(device)
        policy.init_domain_stem(domain, self.hydra_conf.stem)
        policy.init_domain_head(domain, None, self.hydra_conf.head)
        policy.finalize_modules()
        policy.print_model_stats()
        utils.set_seed(self.hydra_conf.seed)

        # add encoders into policy parameters
        if self.hydra_conf.network.finetune_encoder:
            utils.get_image_embeddings(np.zeros((320, 240, 3), dtype=np.uint8), self.hydra_conf.dataset.image_encoder)
            from hpt.utils.utils import global_vision_model
            policy.init_encoders("image", global_vision_model)

        # load the full model
        policy.load_model(os.path.join(self.hydra_conf.train.pretrained_dir, "model.pth"))
        policy.to(device)
        policy.eval()
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"number of params (M): {n_parameters / 1.0e6:.2f}")

    def get_embodiment_proprio_state(self, embodiment='locoman'):
        if embodiment == 'locoman':
            self.body_pose_prorio = self.body_pose_prorio_callback.copy()
            self.eef_pose_proprio = self.eef_pose_proprio_callback.copy()
            self.gripper_angle_proprio = self.gripper_angle_proprio_callback.copy()
            self.joint_pos_proprio = self.joint_pos_proprio_callback.copy()
            self.joint_vel_proprio = self.joint_vel_proprio_callback.copy()

            right_eef_rpy = self.eef_pose_proprio[3:6]
            left_eef_rpy = self.eef_pose_proprio[9:12]
            right_eef_rot_mat = rpy_to_rot_mat(right_eef_rpy)
            left_eef_rot_mat = rpy_to_rot_mat(left_eef_rpy)

            new_right_eef_rot_mat = np.zeros_like(right_eef_rot_mat)
            new_left_eef_rot_mat = np.zeros_like(left_eef_rot_mat)
            new_right_eef_rot_mat[:, 0] = -right_eef_rot_mat[:, 2]
            new_right_eef_rot_mat[:, 1] = right_eef_rot_mat[:, 0]
            new_right_eef_rot_mat[:, 2] = -right_eef_rot_mat[:, 1]
            new_left_eef_rot_mat[:, 0] = -left_eef_rot_mat[:, 2]
            new_left_eef_rot_mat[:, 1] = left_eef_rot_mat[:, 0]
            new_left_eef_rot_mat[:, 2] = -left_eef_rot_mat[:, 1]

            self.eef_pose_proprio[3:6] = rot_mat_to_rpy(new_right_eef_rot_mat)
            self.eef_pose_proprio[9:12] = rot_mat_to_rpy(new_left_eef_rot_mat)

            self.eef_to_body_pose_proprio = self.eef_to_body_pose_proprio.copy()
            self.eef_to_body_pose_proprio[:3] = self.eef_pose_proprio[:3] - self.body_pose_prorio[:3]
            self.eef_to_body_pose_proprio[6:9] = self.eef_pose_proprio[6:9] - self.body_pose_prorio[:3]
            self.eef_to_body_pose_proprio[3:6] = rot_mat_to_rpy(rpy_to_rot_mat(self.body_pose_prorio[3:6]).T @ new_right_eef_rot_mat)
            self.eef_to_body_pose_proprio[9:12] = rot_mat_to_rpy(rpy_to_rot_mat(self.body_pose_prorio[3:6]).T @ new_left_eef_rot_mat)


    def get_embodiment_masks(self, embodiment='locoman'):
        if embodiment == 'locoman':
            # masks support single gripper manipulation mode for now
            self.img_main_mask = np.array([True])
            self.img_wrist_mask = np.array([True])
            # if self.args.use_wrist_camera:
            #     self.img_wrist_mask[self.manipulate_eef_idx] = True
            # though disable the robot body xyz actions for stability, it should be included the proprio info.
            self.proprio_body_mask = np.array([True, True, True, True, True, True])
            # within single gripper manipulation mode: (1) the inactive gripper does not directly have the 6d pose; (2) introduces irrelevant info.
            # even if we would consider bimanual mode, the inactive gripper 6d pose of the single gripper manipulation mode is still less relevant
            self.proprio_eef_mask = np.array([True] * 12)
            # self.proprio_eef_mask = np.array([False] * 12)
            # self.proprio_eef_mask[6*self.manipulate_eef_idx:6+6*self.manipulate_eef_idx] = True
            self.proprio_gripper_mask = np.array([True, True])
            # self.proprio_gripper_mask = np.array([False, False])
            # self.proprio_gripper_mask[self.manipulate_eef_idx] = True
            self.proprio_other_mask = np.array([True, True])
            self.act_body_mask = np.array([False, False, False, True, True, True])
            self.act_eef_mask = np.array([False] * 12)
            self.act_eef_mask[6*self.manipulate_eef_idx:6+6*self.manipulate_eef_idx] = True
            self.act_gripper_mask = np.array([False, False])
            self.act_gripper_mask[self.manipulate_eef_idx] = True

    def transform_eef_pose_trajectory(self, eef_pose_history):
        active_eef_rot_history = eef_pose_history[:, self.manipulate_eef_idx*6+3:self.manipulate_eef_idx*6+6]
        active_eef_rot_mat_history = R.from_euler('xyz', active_eef_rot_history).as_matrix()
        active_eef_rot_mat_history_new = np.zeros_like(active_eef_rot_mat_history)
        active_eef_rot_mat_history_new[:, :, 0] = active_eef_rot_mat_history[:, :, 1]
        active_eef_rot_mat_history_new[:, :, 1] = -active_eef_rot_mat_history[:, :, 2]
        active_eef_rot_mat_history_new[:, :, 2] = -active_eef_rot_mat_history[:, :, 0]
        active_eef_rot_mat_history = active_eef_rot_mat_history_new
        active_eef_rot_history = R.from_matrix(active_eef_rot_mat_history).as_euler('xyz')
        eef_pose_history[:, self.manipulate_eef_idx*6+3:self.manipulate_eef_idx*6+6] = active_eef_rot_history
        return eef_pose_history
    
    def transform_robot_eef_pose_to_uni(self, robot_eef_pose):
        robot_eef_pose_rot_mat = rpy_to_rot_mat(robot_eef_pose[self.manipulate_eef_idx+3:self.manipulate_eef_idx+6])
        robot_eef_pose_rot_mat_uni = np.zeros_like(robot_eef_pose_rot_mat)
        robot_eef_pose_rot_mat_uni[:, 0] = -robot_eef_pose_rot_mat[:, 2]
        robot_eef_pose_rot_mat_uni[:, 1] = robot_eef_pose_rot_mat[:, 0]
        robot_eef_pose_rot_mat_uni[:, 2] = -robot_eef_pose_rot_mat[:, 1]
        robot_eef_pose_rot_uni = rot_mat_to_rpy(robot_eef_pose_rot_mat_uni)
        robot_eef_pose[self.manipulate_eef_idx+3:self.manipulate_eef_idx+6] = robot_eef_pose_rot_uni
        return robot_eef_pose
    
    def temporal_ensemble(self, m):
        # temporal ensemble
        if self.rollout_counter < self.chunk_size:
            self.action_idx = self.rollout_counter
        else:
            self.action_idx = self.action_idx + 1
            if (self.rollout_counter - self.chunk_size) % self.infer_interval == 0:
                self.command_trajs = self.command_trajs[1:]
                self.action_idx = self.action_idx - self.infer_interval
        action = np.zeros_like(self.command)
        weight_sum = 0
        for i in range(len(self.command_trajs)):
            if self.action_idx - i * self.infer_interval >= 0:
                action = action + self.command_trajs[i][self.action_idx - i * self.infer_interval] * np.exp(-m*i)
                weight_sum = weight_sum + np.exp(-m*i)
        action = action / weight_sum
        self.command = action

    def model_inference(self):
        # print('freq', 1 / (time.time() - self.model_time))
        # self.model_time = time.time()
        embodiment = 'locoman'
        self.get_embodiment_proprio_state(embodiment=embodiment)

        ### observations
        obs_dict = {}
        obs_dict['body_pose_state'] = torch.tensor(self.body_pose_prorio).to(self.device).unsqueeze(0)
        obs_dict['eef_pose_state'] = torch.tensor(self.eef_pose_proprio).to(self.device).unsqueeze(0)
        obs_dict['relative_pose_state'] = torch.tensor(self.eef_to_body_pose_proprio).to(self.device).unsqueeze(0)
        obs_dict['gripper_state'] = torch.tensor(self.gripper_angle_proprio).to(self.device).unsqueeze(0)

        if self.hydra_conf.use_real_robot:
            main_image = torch.from_numpy(self.head_color_frame).float()
        else:
            main_image = torch.from_numpy(self.sim_view_frame).float()

        main_image.div_(255.0)
        main_image = torch.einsum('h w c -> c h w', main_image)
        temp = main_image[0].clone()
        main_image[0] = main_image[2]
        main_image[2] = temp
        obs_dict['main_image'] = main_image.unsqueeze(0)

        if self.hydra_conf.use_wrist_camera:
            wrist_image = torch.from_numpy(self.wrist_color_frame).float()
            wrist_image.div_(255.0)
            wrist_image = torch.einsum('h w c -> c h w', wrist_image)
            temp = wrist_image[0].clone()
            wrist_image[0] = wrist_image[2]
            wrist_image[2] = temp
            obs_dict['wrist_image'] = wrist_image.unsqueeze(0)

        ### observation masks
        obs_mask_dict = {}
        obs_mask_dict['main_image'] = torch.tensor(self.img_main_mask).to(self.device)
        if self.hydra_conf.use_wrist_camera:
            obs_mask_dict['wrist_image'] = torch.tensor(self.img_wrist_mask).to(self.device)
        obs_mask_dict['body_pose_state'] = torch.tensor(self.proprio_body_mask).to(self.device).unsqueeze(0)
        obs_mask_dict['eef_pose_state'] = torch.tensor(self.proprio_eef_mask).to(self.device).unsqueeze(0)
        obs_mask_dict['relative_pose_state'] = torch.tensor(self.proprio_eef_mask).to(self.device).unsqueeze(0)
        obs_mask_dict['gripper_state'] = torch.tensor(self.proprio_gripper_mask).to(self.device).unsqueeze(0)
        # delta actions
        # delta_eef: [1, chunk_size, 12]
        # delta_body: [1, chunk_size, 6]
        # delta_gripper: [1, chunk_size, 2]
        
        state_input = torch.cat([obs_dict['body_pose_state'], obs_dict['eef_pose_state'], obs_dict['relative_pose_state'], obs_dict['gripper_state']], dim=1)
        
        
        obs_dict = {"state": state_input, "main_image": obs_dict['main_image'], "wrist_image": obs_dict['wrist_image'], "language_instruction": 'toy collection'} # TODO
        
        self.action = self.bc_policy.get_action(obs_dict)
        
        self.command_traj = self.action.squeeze(0).cpu().detach().numpy()
        
        # self.delta_act_dict = self.bc_model(obs_dict, embodiment, obs_mask_dict)
        

        # if self.args.delta_action:
        #     delta_body_pose_traj = self.delta_act_dict['delta_body_pose'].squeeze(0).cpu().detach().numpy()
        #     delta_eef_pose_traj = self.delta_act_dict['delta_eef_pose'].squeeze(0).cpu().detach().numpy()
        #     delta_gripper_traj = self.delta_act_dict['delta_gripper'].squeeze(0).cpu().detach().numpy()

        #     body_pose_traj = np.cumsum(delta_body_pose_traj, axis=0) + self.robot_reset_pose[:6]
        #     reset_eef_pose_uni = self.transform_robot_eef_pose_to_uni(self.robot_reset_pose[6:18])
        #     eef_pose_traj = np.cumsum(delta_eef_pose_traj, axis=0) + reset_eef_pose_uni
        #     eef_pose_traj = self.transform_eef_pose_trajectory(eef_pose_traj)
        #     gripper_angle_traj = np.cumsum(delta_gripper_traj, axis=0) + self.robot_reset_pose[18:]

        # else:
        #     # hack for now
        #     body_pose_traj = self.delta_act_dict['delta_body_pose'].squeeze(0).cpu().detach().numpy()
        #     eef_pose_traj = self.delta_act_dict['delta_eef_pose'].squeeze(0).cpu().detach().numpy()
        #     eef_pose_traj = self.transform_eef_pose_trajectory(eef_pose_traj)
        #     gripper_angle_traj = self.delta_act_dict['delta_gripper'].squeeze(0).cpu().detach().numpy()

        # self.command_traj = np.concatenate((body_pose_traj * self.act_body_mask,
        #                                     eef_pose_traj * self.act_eef_mask,
        #                                     gripper_angle_traj * self.act_gripper_mask), axis=1)
        self.command_trajs.append(self.command_traj)

        
    def publish_command(self, event=None):
        # publish teleop commands at a fixed rate 
        # need to check the duration to finish one execution and if a separate thread is needed: takes 0.0002-0.0003s
        # print('freq', 1 / (time.time() - self.model_time))
        # self.model_time = time.time()

        self.update_manipulate_eef_idx()
        eef_idx = self.manipulate_eef_idx
        # reset the robot
        if self.begin_to_receive:
            if self.state_flag == 1:
                self.state_flag = 0
                self.robot_reset_publisher.publish(self.robot_reset_msg)
                self.on_reset = True
                self.reset_finished = False
                self.initial_receive = True
                print("reset robot")
        self.command[:] = 0
        
        if self.initial_receive and self.begin_to_receive and self.reset_finished:
            # initialize and start rollout
            if self.state_flag == 2:
                print('start rollout')
                self.state_flag = 0
                if self.teleoperation_mode != 3:
                    # not bi-manual
                    pass
                else:
                    # bi-manual
                    pass
            else:
                return
            
            self.command = self.robot_reset_pose.copy()
            self.command_msg.data = self.command.tolist()
            self.command_publisher.publish(self.command_msg)
            self.initial_receive = False
        # rollout
        elif self.begin_to_receive and self.reset_finished:
            if self.state_flag == 3:
                self.state_flag = 0
                self.pause_commands = True
                print("Pause sending commands")
                return
            if self.pause_commands and self.state_flag == 2:
                self.state_flag = 0
                print("Restart sending commands")
                self.pause_commands = False
            if self.pause_commands:
                return

            # if self.teleoperation_mode != 3:
            #     # not bi-manual
            #     pass
            # else:
            #     # bimanual
            #     pass
            if self.rollout_counter % self.infer_interval == 0:
                self.model_inference()
            self.temporal_ensemble(m=0.1)
            self.rollout_counter += 1
            # self.command = self.command_traj[self.rollout_counter % self.infer_interval]
            self.command_msg.data = self.command.tolist()
            self.command_publisher.publish(self.command_msg)


    def on_press_key(self, key):
        try:
            if self.begin_to_receive:
                if key.char == '1':
                    self.state_flag = 1
            if self.initial_receive and self.begin_to_receive and self.reset_finished:
                if key.char == '2':
                    self.state_flag = 2
            elif self.begin_to_receive and self.reset_finished:
                if key.char == '3':
                    self.state_flag = 3
                if key.char == '2':
                    self.state_flag = 2
        except AttributeError:
            # Handle special keys (like function keys, arrow keys, etc.)
            pass  #
                
    def keyboard_listener_thread(self):
        with keyboard.Listener(on_press=self.on_press_key) as listener:
            listener.join()
    
    def run(self):
        if self.hydra_conf.use_real_robot:
            head_camera_streaming_thread = threading.Thread(target=self.head_camera_stream_thread, daemon=True)
            head_camera_streaming_thread.start()
            if self.hydra_conf.use_wrist_camera:
                wrist_camera_streaming_thread = threading.Thread(target=self.wrist_camera_stream_thread, daemon=True)
                wrist_camera_streaming_thread.start()
        # model_inference_thread = threading.Thread(target=self.model_inference, daemon=True)
        # model_inference_thread.start()
        keyboard_listener_thread = threading.Thread(target=self.keyboard_listener_thread, daemon=True)
        keyboard_listener_thread.start()

        rospy.Timer(rospy.Duration(1.0 / self.hydra_conf.control_freq), self.publish_command)
        rospy.spin()

    def rot_mat_to_rpy_zxy(self, R):
        """
        Convert a rotation matrix (RzRxRy) to Euler angles with ZXY order (first rotate with y, then x, then z).
        This method is more numerically stable, especially near singularities.
        """

        sx = R[2, 1]
        singular_threshold = 1e-6
        cx = np.sqrt(R[2, 0]**2 + R[2, 2]**2)

        if cx < singular_threshold:
            x = np.arctan2(sx, cx)
            y = np.arctan2(R[0, 2], R[0, 0])
            z = 0
        else:
            x = np.arctan2(sx, cx)
            y = np.arctan2(-R[2, 0], R[2, 2])
            z = np.arctan2(-R[0, 1], R[1, 1])

        return np.array([x, y, z])

    def close(self):   
        print('close rollout')

def signal_handler(sig, frame):
    print('pressed Ctrl+C! exiting...')
    rospy.signal_shutdown('Ctrl+C pressed')
    sys.exit(0)

@hydra.main(config_path="../experiments/configs", config_name="config", version_base="1.2")
def main(cfg):
    
    rollout = Rollout(cfg)
    try:
        rollout.run()
    finally:
        rollout.close()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    # parser = argparse.ArgumentParser(prog="teleoperation with apple vision pro and vuer")
    # # command publish rate
    # # - on hand move and on cam move frequence, important
    # parser.add_argument("--use_real_robot", type=str2bool, default=True, help="whether to use real robot.")
    # parser.add_argument("--head_camera_type", type=int, default=1, help="0=realsense, 1=stereo rgb camera")
    # parser.add_argument("--use_wrist_camera", type=str2bool, default=True, help="whether to use wrist camera for real-robot teleop.")
    # parser.add_argument("--desired_stream_fps", type=int, default=60, help="desired camera streaming fps to vuer")
    # parser.add_argument("--control_freq", type=int, default=60, help="control frequency")
    # parser.add_argument("--inference_interval", type=int, default=10, help="model inference interval of publishing commands")
    # parser.add_argument("--action_chunk_size", type=int, default=60, help="action chunk size")
    # parser.add_argument("--device", type=str, default='cuda:0')
    # parser.add_argument("--exp_name", type=str, default='test')
    # parser.add_argument("--delta_action", type=str2bool, default=False, help="if the model output are delta actions")
    # # exp_name

    # args, unknown_args = parser.parse_known_args()
    
    main()









