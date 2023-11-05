import shutil
import tempfile
import urllib.request
from pathlib import Path

import cv2 as cv

from cv_bridge import CvBridge

import numpy as np

import message_filters

import rclpy
from rclpy.node import Node

import sensor_msgs
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from stereo_msgs.msg import DisparityImage

import torch
import torch.nn.functional as F

import time


class DepthStereoNetwork(Node):
    """A ROS2 node for running a deep stereo network."""

    def __init__(self):
        """Initialize the node.
        (1) Declares all parameters
        (2) Load the model depending on the parameters
        (3) Declares the publishers
        (4) Subscribes to the left and right image
        """

        super().__init__('depth_stereo_network')

        self._declare_parameters()

        model_path = Path(
            f'{self._model_name}-{self._device}-{self._height}x{self._width}.scripted.pt'
        )

        if not model_path.exists():
            self._download_model(model_path)
        assert model_path.exists()

        self.net = torch.jit.load(model_path)
        self.net.eval()
        torch.no_grad()
        self.net = self.net.to(torch.device(self._device))

        self._bridge = CvBridge()

        self.pub_disp = self.create_publisher(DisparityImage, '/disparity', 10)
        self.pub_depth = self.create_publisher(Image, '/depth', 10)
        self.pub_pcl = self.create_publisher(PointCloud2, '/points2', 10)

        camera_info_left = message_filters.Subscriber(
            self, CameraInfo, 'left/camera_info'
        )
        image_left = message_filters.Subscriber(self, Image, 'left/image_rect')
        camera_info_right = message_filters.Subscriber(
            self, CameraInfo, 'right/camera_info'
        )
        image_right = message_filters.Subscriber(
            self, Image, 'right/image_rect')

        ts = message_filters.ApproximateTimeSynchronizer(
            [camera_info_left, image_left, camera_info_right, image_right],
            1,
            0.1,
            allow_headerless=True,
        )

        ts.registerCallback(self.callback)
        self.get_logger().info('Node ready.')

    def _declare_parameters(self):
        self._max_depth = (
            self.declare_parameter(
                'max_depth', 20.0).get_parameter_value().double_value
        )
        self._min_depth = (
            self.declare_parameter(
                'min_depth', 0.01).get_parameter_value().double_value
        )
        self._delta_d = (
            self.declare_parameter(
                'delta_d', 1).get_parameter_value().integer_value
        )
        self._width = (
            self.declare_parameter(
                'width', 640).get_parameter_value().integer_value
        )
        self._height = (
            self.declare_parameter(
                'height', 480).get_parameter_value().integer_value
        )
        self._model_name = (
            self.declare_parameter('model_name', 'raft-stereo-fast')
            .get_parameter_value()
            .string_value
        )
        self._device = (
            self.declare_parameter(
                'device', 'cuda').get_parameter_value().string_value
        )
        assert self._device in ['cpu', 'cuda']

    def _download_model(model_path: Path):
        urls = {
            'raft-stereo-fast-cuda-128x160.scripted.pt': 'https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-128x160.scripted.pt',
            'raft-stereo-fast-cuda-256x320.scripted.pt': 'https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-256x320.scripted.pt',
            'raft-stereo-fast-cuda-480x640.scripted.pt': 'https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-480x640.scripted.pt',
            'raft-stereo-fast-cuda-736x1280.scripted.pt': 'https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-736x1280.scripted.pt',
        }
        url = urls[model_path.name]
        filename = model_path.name
        with tempfile.TemporaryDirectory() as d:
            tmp_file_path = Path(d) / filename
            print(f'Downloading {filename} from {url} to {model_path}...')
            urllib.request.urlretrieve(url, tmp_file_path)
            shutil.move(tmp_file_path, model_path)

    def callback(self, msg_cam_info_l, msg_img_l, msg_cam_info_r, msg_img_r):
        """Estimate disparity from a given pair of images and publish the result.
        (1) Preprocess both images and move it to the GPU
        (2) Run the inference
        (3) Move the result back to the CPU and publish as a disparity msg / depth map msg / point cloud msg
        """
        self.get_logger().info(
            f'Image recevied: {msg_img_l.height},{msg_img_l.width} | {msg_img_r.height},{msg_img_r.width}'
        )

        t0 = time.time()
        img_l = self._bridge.imgmsg_to_cv2(msg_img_l)
        img_r = self._bridge.imgmsg_to_cv2(msg_img_r)

        tensor_l = self._preprocess(np.asarray(img_l))
        tensor_r = self._preprocess(np.asarray(img_r))

        t1 = time.time()
        outputs = self.net(tensor_l, tensor_r)
        t2 = time.time()

        disparity = self._postprocess(outputs, target_shape=img_l.shape)

        focal_length = msg_cam_info_l.k[0]
        baseline = -msg_cam_info_l.p[3]/focal_length

        self.pub_disp.publish(self._create_disparity_msg(
            disparity, msg_img_l.header, focal_length=focal_length,
            baseline=baseline))

        # https://github.com/princeton-vl/RAFT-Stereo#converting-disparity-to-depth
        cxl = msg_cam_info_l.k[2]
        cxr = msg_cam_info_r.k[2]
        depth = baseline*focal_length/np.abs(disparity + cxl - cxr)

        self.pub_depth.publish(self._bridge.cv2_to_imgmsg(
            depth, header=msg_img_l.header))

        xyzi = self._reconstruct(
            depth, np.array(msg_cam_info_l.k).reshape(3, 3), img_l)
        self.pub_pcl.publish(self._create_pcl_msg(
            xyzi, msg_img_l.header, 'xyzi'))

        self.get_logger().info(
            f'[{msg_img_l.header.stamp.sec}.{msg_img_l.header.stamp.nanosec}] Total: {time.time() - t0}s, Inference: {t2 - t1}s')

    def _preprocess(self, img: np.ndarray):
        # [H,W,C] -> [H,W,CCC] -> [CCC,H,W]
        # Normalization done in the model itself.
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        img = torch.from_numpy(img).permute(
            2, 0, 1).unsqueeze(0).float().to(torch.device(self._device))
        return F.interpolate(
            img, size=(self._height, self._width), mode='area')

    def _postprocess(self, outputs, target_shape):
        # [N,d,H,W] -> [H,W,d]
        disparity = (
            outputs[1].detach() * -1.0
        )
        if disparity.shape[:2] != target_shape:
            disparity = F.interpolate(disparity, size=(
                target_shape[0], target_shape[1]), mode='area')
            x_scale = target_shape[1] / float(self._width)
            disparity *= np.float32(x_scale)
        disparity = disparity.squeeze().cpu().numpy()
        # mask out image regions with black stripe FIXME
        disparity[:, -40:] = np.nan
        disparity[-20:] = np.nan

    def _reconstruct(self, Z, K, I):
        # xyz = z * K^-1 * uv1
        uv = np.dstack(np.meshgrid(
            np.arange(Z.shape[1]), np.arange(Z.shape[0]))).reshape(-1, 2)
        uv1 = np.ones((uv.shape[0], 3))
        uv1[:, :2] = uv
        xyz = (Z.reshape((-1, 1)) * (np.linalg.inv(K) @
               uv1.T).T).reshape(Z.shape[0], Z.shape[1], 3)
        intensity = I[uv[:, 1], uv[:, 0]].reshape(Z.shape[0], Z.shape[1], 1)
        xyz[xyz[:, :, 2] <= 0] = np.nan
        return np.dstack([xyz, intensity])

    def _create_disparity_msg(self, disparity, header, focal_length, baseline):
        msg_disp_l = DisparityImage()
        msg_disp_l.header = header
        msg_disp_l.image = self._bridge.cv2_to_imgmsg(
            disparity, header=header)

        msg_disp_l.f = focal_length
        msg_disp_l.t = baseline
        msg_disp_l.min_disparity = focal_length * baseline/self._max_depth
        msg_disp_l.max_disparity = focal_length * baseline/self._min_depth
        msg_disp_l.delta_d = self._delta_d

    def _create_pcl_msg(self, points, header, fields='xyz'):
        ros_dtype = sensor_msgs.msg.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = points.astype(dtype).tobytes()

        fields = [sensor_msgs.msg.PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate(fields)]

        return sensor_msgs.msg.PointCloud2(
            header=header,
            height=points.shape[0],
            width=points.shape[1],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * len(fields)),
            row_step=(itemsize * len(fields) * points.shape[1]),
            data=data
        )


def main(args=None):
    """ Main function to create, spin and destroy the node."""
    rclpy.init(args=args)

    node = DepthStereoNetwork()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
