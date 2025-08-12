from std_msgs.msg import String,Float32
from inference import InferenceLaneDetector
from rclpy.node import Node
from std_msgs.msg import Float32
import time
import torch
import numpy as np
from torchvision import transforms as T
from PIL import Image
import math
from Deeplabv3Plus import * 
import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2



class inference(Node):
    def __init__(self):
        super().__init__('inference')
        self.publisher_lat_error = self.create_publisher(Float32, 'nav/lat_error', 10)
        self.publisher_ang_error = self.create_publisher(Float32, 'nav/ang_error', 10)

        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # change topic if needed
            self.listener_callback,
            10)
        self.bridge = CvBridge()

        # Load the PyTorch model once
        self.test_pytorch_model_path = "src/feature_extraction/feature_extraction/deeplab_full_model.pt"
        try:
            self.detector = InferenceLaneDetector(pytorch_model_path=self.test_pytorch_model_path)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize detector: {e}")
            exit()

        self.fixed_size = (512, 512)  # Fixed size for model and display
        self.timer_period = 0.03  # Timer callback period (adjust as needed)
        #self.timer = self.create_timer(self.timer_period, self.listener_callback)
        self.frame_counter = 0  # Counter to keep track of frames

        # EMA Smoothing parameters
        self.smoothed_lat_error = 0.0
        self.smoothed_ang_error = 0.0
        self.alpha = 0.1 



    def listener_callback(self,msg):
        try:
            # Convert ROS Image message to OpenCV image (BGR8 format)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
        
        
        h_orig, w_orig, _ = frame.shape
        # Define the desired ROI size
        roi_width = 700
        roi_height = 800

        # Center the ROI horizontally
        crop_x = max(0, (w_orig - roi_width) // 2)

        # Position the ROI at the bottom vertically
        crop_y = max(0, h_orig - roi_height)

        # Adjust if ROI exceeds image bounds
        crop_w = min(roi_width, w_orig - crop_x)
        crop_h = min(roi_height, h_orig - crop_y)

        # Crop the frame
        frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        # Resize frame for model input
        frame_resized = cv2.resize(frame, self.fixed_size)
        display_frame = frame_resized.copy()

        # Run inference
        start_time = time.time()
        avg_left, avg_right, nav_line, raw_lat_error, raw_ang_error = self.detector.calculate(frame_resized)
        end_time = time.time()
        # Apply Exponential Moving Average (EMA)
        if self.frame_counter == 0: # Initialize smoothed values on the very first processed frame
            self.smoothed_lat_error = raw_lat_error
            self.smoothed_ang_error = raw_ang_error
        else:
            self.smoothed_lat_error = self.alpha * raw_lat_error + (1 - self.alpha) * self.smoothed_lat_error
            self.smoothed_ang_error = self.alpha * raw_ang_error + (1 - self.alpha) * self.smoothed_ang_error

        # Use smoothed errors for display and publishing
        lat_error = self.smoothed_lat_error
        ang_error = self.smoothed_ang_error

        # Draw lanes and nav line
        if avg_left:
            cv2.line(display_frame, avg_left[0], avg_left[1], (255, 255, 0), 2)
        if avg_right:
            cv2.line(display_frame, avg_right[0], avg_right[1], (255, 0, 255), 2)
        if nav_line:
            cv2.line(display_frame, nav_line[0], nav_line[1], (0, 255, 0), 3)

        # Draw center reference
        cv2.line(display_frame, (self.fixed_size[0] // 2, 0), (self.fixed_size[0] // 2, self.fixed_size[1] - 1), (0, 0, 255), 1)

        # Overlay latency + errors
        fps = 1.0 / (end_time - start_time)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Lat Error: {lat_error:.1f}px", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Ang Error: {ang_error:.1f} deg", (10, 490),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the result
        cv2.imshow("Lane Detection (512x512)", display_frame)

        # Create the Float32 messages
        lat_error_msg = Float32()
        lat_error_msg.data = lat_error  # Assign the value of lat_error

        ang_error_msg = Float32()
        ang_error_msg.data = ang_error  # Assign the value of ang_error

        # Publish the messages
        self.publisher_lat_error.publish(lat_error_msg)
        self.publisher_ang_error.publish(ang_error_msg)

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()  # Release video capture when exiting
            cv2.destroyAllWindows()
            return  # Stop processing
        


            
                



def main(args=None):
    rclpy.init(args=args)
    node = inference()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()
        
        
