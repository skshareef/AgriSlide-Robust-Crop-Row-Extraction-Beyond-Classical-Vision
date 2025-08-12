import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# Assuming your classes are in a file named 'ml_classical_pipeline.py'
# in the same package or accessible via your Python path.
from app2 import InferenceLaneDetector, ClassicalLaneProcessor
from Deeplabv3Plus import *

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.get_logger().info('Lane Detection Node with Advanced Visualization has started.')

        # --- Publishers ---
        self.publisher_lat_error = self.create_publisher(Float32, 'nav/lat_error', 10)
        self.publisher_ang_error = self.create_publisher(Float32, 'nav/ang_error', 10)

        # --- Subscription ---
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()

        # --- Parameters ---
        processing_width = 512
        processing_height = 512
        self.fixed_size = (processing_width, processing_height)
        self.frame_counter = 0

        # --- EMA Smoothing Parameters ---
        self.smoothed_lat_error = 0.0
        self.smoothed_ang_error = 0.0
        self.alpha = 0.1

        # --- Initialize Detectors ONCE (Critical for Performance) ---
        pytorch_model_path = "src/feature_extraction/feature_extraction/deeplab_full_model.pt"
        try:
            self.ml_detector = InferenceLaneDetector(pytorch_model_path=pytorch_model_path,
                                                     model_input_width=processing_width,
                                                     model_input_height=processing_height)
            self.classical_processor = ClassicalLaneProcessor(img_width=processing_width, img_height=processing_height)
            self.get_logger().info('Detectors initialized successfully.')
        except Exception as e:
            self.get_logger().error(f"FATAL: Failed to initialize detectors: {e}")
            rclpy.shutdown()

    def listener_callback(self, msg):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # --- ROI Cropping (Optional, but kept from your original code) ---
        h_orig, w_orig, _ = cv_frame.shape
        roi_width = 700
        roi_height = 800
        crop_x = max(0, (w_orig - roi_width) // 2)
        crop_y = max(0, h_orig - roi_height)
        cropped_frame = cv_frame[crop_y:crop_y + roi_height, crop_x:crop_x + roi_width]
        
        # --- Resize for Processing & Create Main Display Copy ---
        frame_resized = cv2.resize(cropped_frame, self.fixed_size)
        display_frame = frame_resized.copy() # Main view for drawing final lanes

        start_time = time.time()

        # --- PIPELINE EXECUTION ---
        binary_mask = self.ml_detector.get_mask(frame_resized)
        # Unpack all visualization frames and data from the classical processor
        detected_poly_lines, nav_line, viz_img, cleaned_mask, lateral_error, angular_error = self.classical_processor.find_lanes_and_error(binary_mask)

        end_time = time.time()

        # --- Apply EMA Smoothing ---
        if self.frame_counter == 0:
            self.smoothed_lat_error = lateral_error
            self.smoothed_ang_error = angular_error
        else:
            self.smoothed_lat_error = self.alpha * lateral_error + (1 - self.alpha) * self.smoothed_lat_error
            self.smoothed_ang_error = self.alpha * angular_error + (1 - self.alpha) * self.smoothed_ang_error
        self.frame_counter += 1

        # --- Publish Smoothed Error Values ---
        lat_error_msg = Float32()
        lat_error_msg.data = self.smoothed_lat_error
        self.publisher_lat_error.publish(lat_error_msg)
        ang_error_msg = Float32()
        ang_error_msg.data = self.smoothed_ang_error
        self.publisher_ang_error.publish(ang_error_msg)

        # --- BUILD ADVANCED VISUALIZATION ---

        # 1. Draw final results on the main display frame
        if detected_poly_lines:
            for line_pts in detected_poly_lines:
                cv2.polylines(display_frame, [line_pts.astype(np.int32)], isClosed=False, color=(255, 255, 0), thickness=3)
        if nav_line:
            cv2.line(display_frame, nav_line[0], nav_line[1], (0, 255, 0), 2)
        center_x = self.fixed_size[0] // 2
        cv2.line(display_frame, (center_x, 0), (center_x, self.fixed_size[1] - 1), (0, 0, 255), 2)

        # 2. Prepare the small visualization panes (mask and sliding windows)
        if viz_img is not None:
            classical_viz_resized = cv2.resize(viz_img, (self.fixed_size[0] // 2, self.fixed_size[1] // 2))
        else:
            classical_viz_resized = np.zeros((self.fixed_size[1] // 2, self.fixed_size[0] // 2, 3), dtype=np.uint8)
        
        mask_viz = np.dstack((cleaned_mask, cleaned_mask, cleaned_mask))
        mask_viz_resized = cv2.resize(mask_viz, (self.fixed_size[0] // 2, self.fixed_size[1] // 2))

        # 3. Combine the small panes and add padding
        top_row_viz = np.concatenate((mask_viz_resized, classical_viz_resized), axis=1)
        padding_height = self.fixed_size[1] - top_row_viz.shape[0]
        padding = np.zeros((padding_height, top_row_viz.shape[1], 3), dtype=np.uint8)
        combined_viz = np.concatenate((top_row_viz, padding), axis=0)

        # 4. Combine the main display with the visualization panes
        final_display = np.concatenate((display_frame, combined_viz), axis=1)

        # 5. Add text (FPS, errors) to the final composite image
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(final_display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(final_display, f"Lat Error: {self.smoothed_lat_error:.2f} px", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(final_display, f"Ang Error: {self.smoothed_ang_error:.2f} deg", (10, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Show the final composite image ---
        cv2.imshow("Hybrid Lane Detection", final_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Shutting down node.')
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
