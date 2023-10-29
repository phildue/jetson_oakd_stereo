import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
import message_filters
from cv_bridge import CvBridge
import torch
from pathlib import Path
import urllib.request
import shutil
import tempfile
import cv2 as cv
import numpy as np
urls = {
    "raft-stereo-eth3d-cpu-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cpu-128x160.scripted.pt",
    "raft-stereo-eth3d-cpu-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cpu-256x320.scripted.pt",
    "raft-stereo-eth3d-cpu-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cpu-480x640.scripted.pt",
    "raft-stereo-eth3d-cpu-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cpu-736x1280.scripted.pt",
    "raft-stereo-eth3d-cuda-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cuda-128x160.scripted.pt",
    "raft-stereo-eth3d-cuda-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cuda-256x320.scripted.pt",
    "raft-stereo-eth3d-cuda-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cuda-480x640.scripted.pt",
    "raft-stereo-eth3d-cuda-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-eth3d-cuda-736x1280.scripted.pt",
    "raft-stereo-fast-cpu-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cpu-128x160.scripted.pt",
    "raft-stereo-fast-cpu-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cpu-256x320.scripted.pt",
    "raft-stereo-fast-cpu-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cpu-480x640.scripted.pt",
    "raft-stereo-fast-cpu-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cpu-736x1280.scripted.pt",
    "raft-stereo-fast-cuda-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-128x160.scripted.pt",
    "raft-stereo-fast-cuda-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-256x320.scripted.pt",
    "raft-stereo-fast-cuda-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-480x640.scripted.pt",
    "raft-stereo-fast-cuda-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-fast-cuda-736x1280.scripted.pt",
    "raft-stereo-middlebury-cpu-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cpu-128x160.scripted.pt",
    "raft-stereo-middlebury-cpu-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cpu-256x320.scripted.pt",
    "raft-stereo-middlebury-cpu-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cpu-480x640.scripted.pt",
    "raft-stereo-middlebury-cpu-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cpu-736x1280.scripted.pt",
    "raft-stereo-middlebury-cuda-128x160.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cuda-128x160.scripted.pt",
    "raft-stereo-middlebury-cuda-256x320.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cuda-256x320.scripted.pt",
    "raft-stereo-middlebury-cuda-480x640.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cuda-480x640.scripted.pt",
    "raft-stereo-middlebury-cuda-736x1280.scripted.pt": "https://github.com/nburrus/stereodemo/releases/download/v0.1-raft-stereo/raft-stereo-middlebury-cuda-736x1280.scripted.pt",
}


def download_model (url: str, model_path: Path):
    filename = model_path.name
    with tempfile.TemporaryDirectory() as d:
        tmp_file_path = Path(d) / filename
        print (f"Downloading {filename} from {url} to {model_path}...")
        urllib.request.urlretrieve(url, tmp_file_path)
        shutil.move (tmp_file_path, model_path)


class DepthStereoNetwork(Node):

    def __init__(self):
        super().__init__('depth_stereo_network')
        self._width = self.declare_parameter("width", 640).get_parameter_value().integer_value
        self._height = self.declare_parameter("height", 480).get_parameter_value().integer_value
        self._model_name = self.declare_parameter("model_name", "raft-stereo-fast").get_parameter_value().string_value
        self._device = self.declare_parameter("device", "cuda").get_parameter_value().string_value
        assert self._device in ["cpu", "cuda"]
        model_path = Path(f"{self._model_name}-{self._device}-{self._height}x{self._width}.scripted.pt")
       
        if not model_path.exists():
            #TODO manage this in share directory ?
            download_model(urls[model_path.name], model_path)

        assert model_path.exists()
        self.net = torch.jit.load(model_path)
        self.net.eval()
        torch.no_grad()
        self.net = self.net.to(torch.device(self._device))

        self.pub = self.create_publisher(DisparityImage, "/left/disparity", 10)
        self.pub_img = self.create_publisher(Image, "/left/disparity/image", 10)

        camera_info_left = message_filters.Subscriber(self,CameraInfo,"left/camera_info")
        image_left = message_filters.Subscriber(self,Image,"left/image_rect")
        camera_info_right = message_filters.Subscriber(self,CameraInfo,"right/camera_info")
        image_right = message_filters.Subscriber(self,Image,"right/image_rect")

        ts = message_filters.ApproximateTimeSynchronizer([camera_info_left, image_left, camera_info_right, image_right], 1, 0.1, allow_headerless=True)
        ts.registerCallback(self.estimate_depth)
        print("Node ready.")

    def estimate_depth(self, msg_cam_info_l, msg_img_l, msg_cam_info_r, msg_img_r):
        self.get_logger().info(f'Image recevied: {msg_img_l.height},{msg_img_l.width} | {msg_img_r.height},{msg_img_r.width}')
        bridge = CvBridge()
        img_l = bridge.imgmsg_to_cv2(msg_img_l)
        img_r = bridge.imgmsg_to_cv2(msg_img_r)


        self.get_logger().info('Preprocessing')
        #TODO: we can do this on the gpu too
        left_tensor = self._preprocess_input(np.asarray(img_l))
        right_tensor = self._preprocess_input(np.asarray(img_r))

        self.get_logger().info('To Gpu')
        device = torch.device(self._device)

        left_tensor = left_tensor.to(device)
        right_tensor = right_tensor.to(device)

        self.get_logger().info('Inference')
        outputs = self.net(left_tensor, right_tensor)

        self.get_logger().info('Postprocess..')
        #TODO: we can do this on the gpu too
        disparity = self._process_output(outputs)
        if disparity.shape[:2] != img_l.shape:
            disparity = cv.resize(disparity, (img_l.shape[1], img_r.shape[0]), cv.INTER_NEAREST)
            x_scale = img_l.shape[1] / float(self._width)
            disparity *= np.float32(x_scale)

        msg_disp_l = DisparityImage()
        msg_disp_l.header = msg_img_l.header
        msg_disp_l.image = bridge.cv2_to_imgmsg(disparity,header=msg_img_l.header)
        msg_disp_l.f = msg_cam_info_l.k[0]
        msg_disp_l.t = msg_cam_info_r.p[3]
        msg_disp_l.min_disparity = 1.0 #?
        msg_disp_l.max_disparity = float(img_l.shape[1])
        self.pub.publish(msg_disp_l)
        self.pub_img.publish(msg_disp_l.image)


    def _preprocess_input (self, img: np.ndarray):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (self._width, self._height), cv.INTER_AREA)
        # -> C,H,W
        # Normalization done in the model itself.
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    def _process_output(self, outputs):
        disparity_map = outputs[1][0].detach().cpu().squeeze(0).squeeze(0).numpy() * -1.0
        return disparity_map
def main(args=None):
    rclpy.init(args=args)

    node = DepthStereoNetwork()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
