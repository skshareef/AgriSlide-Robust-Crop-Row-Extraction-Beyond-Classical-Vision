# ml_classical_pipeline.py

import torch
import cv2
import time
import numpy as np
from torchvision import transforms as T
from PIL import Image
import math
from collections import deque

# Assuming these files are in the same directory or accessible in the python path
from Deeplabv3Plus import *

class InferenceLaneDetector:
    """
    This class uses a Deep Learning model (Deeplabv3+) to perform semantic segmentation,
    producing a binary mask of the lanes in an image.
    """
    def __init__(self,
                 pytorch_model_path,
                 model_input_width=512,
                 model_input_height=512,
                 device=None):

        self.model_input_width = model_input_width
        self.model_input_height = model_input_height

        if device:
            self.pytorch_device = torch.device(device)
        else:
            self.pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- LOAD PYTORCH MODEL ---
        print(f"Loading PyTorch model from: {pytorch_model_path}")
        try:
            self.model = torch.load(pytorch_model_path, map_location=self.pytorch_device)
            if not isinstance(self.model, torch.nn.Module):
                 raise TypeError("Loaded object is not a nn.Module")
            print("Successfully loaded full model object.")
        except Exception as e1:
            print(f"Failed to load model directly: {e1}")
            print("Attempting to load state_dict assuming model class is Deeplabv3Plus(num_classes=1).")
            try:
                self.model = Deeplabv3Plus(num_classes=1)
                self.model.load_state_dict(torch.load(pytorch_model_path, map_location=self.pytorch_device))
                print("Successfully loaded state_dict into Deeplabv3Plus(num_classes=1) instance.")
            except Exception as e2:
                print(f"Failed to load state_dict: {e2}")
                raise RuntimeError("Could not load the PyTorch model.")

        self.model.to(self.pytorch_device)
        self.model.eval()
        print("PyTorch model loaded and set to evaluation mode on device:", self.pytorch_device)

        # --- IMAGE TRANSFORM ---
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_mask(self, image):
        """
        Takes a BGR image, runs inference, and returns a binary mask.
        """
        rgb_frame_for_model = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame_for_model)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.pytorch_device)

        with torch.no_grad():
            logits = self.model(input_tensor)

        if isinstance(logits, dict):
            logits = logits.get('out', next((l for l in logits.values() if isinstance(l, torch.Tensor) and l.ndim == 4), None))
        if logits is None:
            raise ValueError("Model output issue: Could not find a valid tensor in the model's output dictionary.")

        logits_for_mask = logits[:, 0:1, :, :]
        probs_gpu = torch.sigmoid(logits_for_mask)
        binary_mask_gpu_tensor = (probs_gpu[0, 0] > 0.4).to(torch.uint8) * 255
        mask_np_uint8 = binary_mask_gpu_tensor.cpu().numpy()

        return mask_np_uint8

class ClassicalLaneProcessor:
    """
    This class takes a binary mask and uses classical CV techniques (sliding windows)
    to detect and fit lines to the lanes.
    """
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height
        
        self.tracker_params = {
            "num_windows": 10,
            "margin": 20,
            "minpix": 70
        }
        
        # State variables for tracking and smoothing across frames
        self.detected = False
        self.left_fit = None
        self.right_fit = None
        self.recent_left_fits = deque(maxlen=5)
        self.recent_right_fits = deque(maxlen=5)

    # ---- outline-only drawer for per-window rectangles (viz only) ----
    def _draw_window_box(self, canvas, x_center, y_low, y_high, margin, edge_color, label):
        """
        Draw an outline-only rectangle centered at x_center with vertical span [y_low, y_high],
        using the actual tracker margin. Adds a label on the top-left corner.
        """
        h, w = canvas.shape[:2]
        x1 = max(0, int(x_center - margin))
        x2 = min(w - 1, int(x_center + margin))
        y1 = max(0, int(y_low))
        y2 = min(h - 1, int(y_high))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), edge_color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, label, (x1 + 4, max(14, y1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas
    # ------------------------------------------------------------------

    def find_lanes_and_error(self, binary_mask):
        """
        Processes a binary mask to find lane lines and calculates the angular and lateral errors.
        Returns:
          final_lines_poly, nav_line_straight, viz_img, cleaned_mask, angular_error_deg, lateral_error_px
        """
        kernel = np.ones((7, 7), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        if not self.detected:
            # Perform a full search if we haven't detected lanes yet
            leftx, lefty, rightx, righty, viz_img = self.sliding_window_search(cleaned_mask)
        else:
            # Otherwise, search in a margin around the previously found line
            leftx, lefty, rightx, righty, viz_img = self.search_around_poly(cleaned_mask)

        # --- Fit Polynomials ---
        left_fit_success, right_fit_success = False, False
        if len(lefty) > self.tracker_params['minpix']:
            try:
                self.left_fit = np.polyfit(lefty, leftx, 2)
                left_fit_success = True
            except np.linalg.LinAlgError:
                print("Warning: Left polyfit failed.")
                self.left_fit = None

        if len(righty) > self.tracker_params['minpix']:
            try:
                self.right_fit = np.polyfit(righty, rightx, 2)
                right_fit_success = True
            except np.linalg.LinAlgError:
                print("Warning: Right polyfit failed.")
                self.right_fit = None
        
        # --- Sanity Check and Smoothing ---
        if left_fit_success and right_fit_success:
            ploty = np.linspace(0, self.img_height-1, self.img_height)
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
            
            lane_width = np.mean(right_fitx - left_fitx)
            if lane_width > 100:
                self.detected = True
                self.recent_left_fits.append(self.left_fit)
                self.recent_right_fits.append(self.right_fit)
            else:
                self.detected = False
        else:
            self.detected = False

        if self.recent_left_fits:
            avg_left_fit = np.mean(self.recent_left_fits, axis=0)
            avg_right_fit = np.mean(self.recent_right_fits, axis=0)
        else:
            avg_left_fit = self.left_fit
            avg_right_fit = self.right_fit

        # --- Generate Final Lines and Calculate Errors ---
        final_lines_poly = []
        angular_error = 0.0
        lateral_error = 0.0
        nav_line_straight = None
        
        if avg_left_fit is not None and avg_right_fit is not None:
            ploty = np.linspace(0, self.img_height-1, self.img_height)
            
            # Generate points for the curved polynomial lines
            left_fitx = avg_left_fit[0]*ploty**2 + avg_left_fit[1]*ploty + avg_left_fit[2]
            left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            final_lines_poly.append(left_line_pts)
            
            right_fitx = avg_right_fit[0]*ploty**2 + avg_right_fit[1]*ploty + avg_right_fit[2]
            right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
            final_lines_poly.append(right_line_pts)

            # Calculate midline and angle from polynomial endpoints
            left_x_bottom, right_x_bottom = left_fitx[-1], right_fitx[-1]
            left_x_top, right_x_top = left_fitx[0], right_fitx[0]

            nav_bottom_x = (left_x_bottom + right_x_bottom) / 2
            nav_top_x = (left_x_top + right_x_top) / 2
            nav_line_straight = ((int(nav_bottom_x), self.img_height -1), (int(nav_top_x), 0))
            
            # --- Angular Error ---
            dx = nav_top_x - nav_bottom_x
            dy = 0 - (self.img_height - 1)
            angular_error = math.degrees(math.atan2(dx, -dy))

            # --- Lateral Error ---
            lateral_error = nav_bottom_x - (self.img_width / 2)

        return final_lines_poly, nav_line_straight, viz_img, cleaned_mask, angular_error, lateral_error

    def sliding_window_search(self, cleaned_mask):
        out_img = np.dstack((cleaned_mask, cleaned_mask, cleaned_mask))
        histogram = np.sum(cleaned_mask[self.img_height//2:,:], axis=0)
        midpoint = int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = self.tracker_params["num_windows"]
        window_height = int(self.img_height/nwindows)
        
        nonzero = cleaned_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = self.tracker_params["margin"]
        minpix = self.tracker_params["minpix"]

        left_lane_inds = []
        right_lane_inds = []

        # For educational overlay: store window centers to connect
        left_centers = []
        right_centers = []

        for window in range(nwindows):
            win_y_low = self.img_height - (window+1)*window_height
            win_y_high = self.img_height - window*window_height
            y_center = (win_y_low + win_y_high) / 2.0

            # draw outline-only rectangles (no fill), using actual margin
            out_img = self._draw_window_box(out_img, leftx_current, win_y_low, win_y_high,
                                            margin, (0, 180, 0), f"L{window}")
            out_img = self._draw_window_box(out_img, rightx_current, win_y_low, win_y_high,
                                            margin, (180, 0, 0), f"R{window}")

            # store centers BEFORE updating (center used for this window)
            left_centers.append((int(leftx_current), int(y_center)))
            right_centers.append((int(rightx_current), int(y_center)))

            # collect indices using the actual margin
            win_xleft_low  = leftx_current  - margin
            win_xleft_high = leftx_current  + margin
            win_xright_low  = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # update current positions if enough pixels found
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # connect the window centers to show how lines are formed
        if len(left_centers) >= 2:
            pts_left = np.array(left_centers, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out_img, [pts_left], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            for (cx, cy) in left_centers:
                cv2.circle(out_img, (cx, cy), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        if len(right_centers) >= 2:
            pts_right = np.array(right_centers, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out_img, [pts_right], isClosed=False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            for (cx, cy) in right_centers:
                cv2.circle(out_img, (cx, cy), 3, (255, 255, 0), -1, lineType=cv2.LINE_AA)

        # legend
        cv2.putText(out_img, "Sliding Window Viz: outline boxes, centers, tracks",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # pack points
        if left_lane_inds and any(ind.size > 0 for ind in left_lane_inds):
            left_lane_inds = np.concatenate(left_lane_inds)
        else:
            left_lane_inds = np.array([], dtype=int)

        if right_lane_inds and any(ind.size > 0 for ind in right_lane_inds):
            right_lane_inds = np.concatenate(right_lane_inds)
        else:
            right_lane_inds = np.array([], dtype=int)
        
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        # Highlight the selected pixels (kept from your original viz)
        out_img[lefty, leftx] = [255, 0, 0]   # left points -> blue channel
        out_img[righty, rightx] = [0, 0, 255] # right points -> red channel

        return leftx, lefty, rightx, righty, out_img

    def search_around_poly(self, cleaned_mask):
        margin = self.tracker_params["margin"]
        nonzero = cleaned_mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_x = self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2]
        right_lane_x = self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2]

        left_lane_inds = ((nonzerox > (left_lane_x - margin)) & (nonzerox < (left_lane_x + margin)))
        right_lane_inds = ((nonzerox > (right_lane_x - margin)) & (nonzerox < (right_lane_x + margin)))

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
        
        # base visualization canvas
        out_img = np.dstack((cleaned_mask, cleaned_mask, cleaned_mask))

        # highlight selected pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # ---------- DISCRETE OUTLINE BOXES along the previous polynomials ----------
        h = cleaned_mask.shape[0]
        nwindows = self.tracker_params["num_windows"]
        window_height = int(h / nwindows)

        for k in range(nwindows):
            y_low  = h - (k + 1) * window_height
            y_high = h - k * window_height
            y_mid  = (y_low + y_high) / 2.0

            # predicted centers from previous fits
            x_left_mid  = self.left_fit[0]*(y_mid**2)  + self.left_fit[1]*y_mid  + self.left_fit[2]
            x_right_mid = self.right_fit[0]*(y_mid**2) + self.right_fit[1]*y_mid + self.right_fit[2]

            # outline-only rectangles (no fill)
            out_img = self._draw_window_box(out_img, int(x_left_mid),  y_low, y_high, margin, (0, 180, 0),  f"L{k}")
            out_img = self._draw_window_box(out_img, int(x_right_mid), y_low, y_high, margin, (180, 0, 0), f"R{k}")

        # legend
        cv2.putText(out_img, "Search-around-poly: outline boxes (+/- margin)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        # --------------------------------------------------------------------------

        return leftx, lefty, rightx, righty, out_img

# --- Main function for testing the new hybrid pipeline ---
if __name__ == '__main__':
    print("--- ML-Classical Hybrid Lane Detection Pipeline Test ---")

    # --- CONFIGURATION ---
    test_pytorch_model_path = "src/feature_extraction/feature_extraction/deeplab_full_model.pt"
    test_video_path = "src/feature_extraction/feature_extraction/1920 x 1080 test4 10 7 24.mp4"
    
    # Define the processing size
    processing_width = 512
    processing_height = 512
    fixed_size = (processing_width, processing_height)

    # --- INITIALIZATION ---
    cap = cv2.VideoCapture(test_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {test_video_path}")
        exit()

    try:
        ml_detector = InferenceLaneDetector(pytorch_model_path=test_pytorch_model_path, 
                                            model_input_width=processing_width,
                                            model_input_height=processing_height)
        
        classical_processor = ClassicalLaneProcessor(img_width=processing_width, img_height=processing_height)
    except Exception as e:
        print(f"Failed to initialize detectors: {e}")
        exit()

    # --- PREPARE THREE SAME-SIZE WINDOWS ---
    cv2.namedWindow("Lane Overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Overlay", fixed_size[0], fixed_size[1])
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", fixed_size[0], fixed_size[1])
    cv2.namedWindow("Sliding Window Viz", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sliding Window Viz", fixed_size[0], fixed_size[1])

    # --- VIDEO LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error.")
            break
        
        h_orig, w_orig, _ = frame.shape
        # Define the desired ROI size
        roi_width = 1000
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
        frame_resized = cv2.resize(frame, fixed_size)
        display_frame = frame_resized.copy()

        start_time = time.time()

        # --- PIPELINE EXECUTION ---
        binary_mask = ml_detector.get_mask(frame_resized)
        detected_poly_lines, nav_line, viz_img, cleaned_mask, angular_error, lateral_error = classical_processor.find_lanes_and_error(binary_mask)
        
        end_time = time.time()
        fps = 1.0 / max(1e-6, (end_time - start_time))
        
        # --- VISUALIZATION (THREE WINDOWS) ---
        # 1) Lane Overlay
        overlay_frame = display_frame.copy()
        if detected_poly_lines:
            for line_pts in detected_poly_lines:
                cv2.polylines(overlay_frame, [line_pts.astype(np.int32)], isClosed=False, color=(255,255,0), thickness=3) # Cyan for poly lines

        if nav_line:
            cv2.line(overlay_frame, nav_line[0], nav_line[1], (0, 255, 0), 2) # Green for nav line

        center_x = processing_width // 2
        cv2.line(overlay_frame, (center_x, 0), (center_x, processing_height - 1), (0, 0, 255), 2) # Red for center line

        cv2.putText(overlay_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(overlay_frame, f"Lateral Error: {lateral_error:.2f} px", (10, processing_height - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay_frame, f"Angular Error: {angular_error:.2f} deg", (10, processing_height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        overlay_frame = cv2.resize(overlay_frame, fixed_size)

        # 2) Mask Window
        mask_viz = np.dstack((cleaned_mask, cleaned_mask, cleaned_mask))
        mask_viz = cv2.resize(mask_viz, fixed_size)

        # 3) Sliding Window Viz
        classical_viz = viz_img if viz_img is not None else np.zeros((fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
        classical_viz = cv2.resize(classical_viz, fixed_size)

        # Show three windows
        cv2.imshow("Lane Overlay", overlay_frame)
        cv2.imshow("Mask", mask_viz)
        cv2.imshow("Sliding Window Viz", classical_viz)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- CLEANUP ---
    cap.release()
    cv2.destroyAllWindows()
    print("--- Video Processing Finished ---")
