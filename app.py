import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

class VirtualMouse:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Camera dimensions (will be set when camera starts)
        self.cam_width = 0
        self.cam_height = 0
        
        # Enhanced movement settings
        self.smoothing = 3  # Reduced for more responsive movement
        self.prev_x, self.prev_y = 0, 0
        
        # Movement area boundaries (use only center portion of camera)
        self.movement_area = {
            'left': 0.2,    # 20% from left
            'right': 0.8,   # 80% from left (20% from right)
            'top': 0.2,     # 20% from top
            'bottom': 0.8   # 80% from top (20% from bottom)
        }
        
        # Sensitivity multiplier
        self.sensitivity = 1.5
        
        # Click detection variables
        self.click_threshold = 40
        self.right_click_performed = False
        
        # Hold/Drag functionality variables
        self.left_click_start_time = None
        self.left_click_held = False
        self.hold_duration_threshold = 0.4 # seconds
        
    def get_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def smooth_mouse_movement(self, x, y):
        """Enhanced smooth mouse movement with acceleration"""
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = x, y
            return x, y
        
        # Calculate movement distance for acceleration
        distance = math.sqrt((x - self.prev_x)**2 + (y - self.prev_y)**2)
        
        # Apply acceleration based on movement distance
        if distance > 50:  # Large movement
            acceleration = 1.2
        elif distance > 20:  # Medium movement
            acceleration = 1.0
        else:  # Small movement
            acceleration = 0.8
        
        smooth_x = self.prev_x + (x - self.prev_x) / self.smoothing * acceleration
        smooth_y = self.prev_y + (y - self.prev_y) / self.smoothing * acceleration
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)
    
    def map_coordinates(self, finger_x, finger_y):
        """Map finger coordinates to screen coordinates with enhanced sensitivity"""
        # Define the movement area boundaries in camera coordinates
        left_bound = self.cam_width * self.movement_area['left']
        right_bound = self.cam_width * self.movement_area['right']
        top_bound = self.cam_height * self.movement_area['top']
        bottom_bound = self.cam_height * self.movement_area['bottom']
        
        # Clamp finger position to movement area
        clamped_x = max(left_bound, min(right_bound, finger_x))
        clamped_y = max(top_bound, min(bottom_bound, finger_y))
        
        # Map to screen coordinates with enhanced sensitivity
        screen_x = np.interp(clamped_x, [left_bound, right_bound], [0, self.screen_width])
        screen_y = np.interp(clamped_y, [top_bound, bottom_bound], [0, self.screen_height])
        
        # Apply sensitivity multiplier
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        
        # Calculate offset from center and apply sensitivity
        offset_x = (screen_x - center_x) * self.sensitivity
        offset_y = (screen_y - center_y) * self.sensitivity
        
        # Calculate final position
        final_x = center_x + offset_x
        final_y = center_y + offset_y
        
        # Clamp to screen boundaries
        final_x = max(0, min(self.screen_width - 1, final_x))
        final_y = max(0, min(self.screen_height - 1, final_y))
        
        return final_x, final_y
    
    def process_frame(self, frame):
        """Process each frame for hand detection and gesture recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Draw movement area rectangle
        left_bound = int(self.cam_width * self.movement_area['left'])
        right_bound = int(self.cam_width * self.movement_area['right'])
        top_bound = int(self.cam_height * self.movement_area['top'])
        bottom_bound = int(self.cam_height * self.movement_area['bottom'])
        
        cv2.rectangle(frame, (left_bound, top_bound), (right_bound, bottom_bound), 
                      (255, 255, 0), 2)
        cv2.putText(frame, "MOVE AREA", (left_bound + 10, top_bound - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * self.cam_width)
                    y = int(lm.y * self.cam_height)
                    landmarks.append([x, y])
                
                # Get specific finger tip positions
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                
                # Enhanced mouse movement using index finger
                screen_x, screen_y = self.map_coordinates(index_tip[0], index_tip[1])
                
                # Apply smoothing
                smooth_x, smooth_y = self.smooth_mouse_movement(screen_x, screen_y)
                
                # Move mouse
                pyautogui.moveTo(smooth_x, smooth_y)
                
                # Enhanced visual feedback for index finger
                cv2.circle(frame, tuple(index_tip), 12, (0, 255, 0), -1)
                cv2.circle(frame, tuple(index_tip), 20, (0, 255, 0), 2)
                
                # Show if finger is in movement area
                if (left_bound <= index_tip[0] <= right_bound and 
                    top_bound <= index_tip[1] <= bottom_bound):
                    cv2.putText(frame, "ACTIVE", (index_tip[0] - 30, index_tip[1] - 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "OUT OF AREA", (index_tip[0] - 50, index_tip[1] - 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Left click / hold detection (thumb + middle finger)
                thumb_middle_dist = self.get_distance(thumb_tip, middle_tip)

                if thumb_middle_dist < self.click_threshold:
                    if self.left_click_start_time is None:
                        self.left_click_start_time = time.time()
                    
                    hold_time = time.time() - self.left_click_start_time

                    if hold_time > self.hold_duration_threshold and not self.left_click_held:
                        self.left_click_held = True
                        pyautogui.mouseDown()
                        print("Left Mouse Down (Hold)")

                    if self.left_click_held:
                        cv2.putText(frame, "HOLDING", (thumb_tip[0] - 50, thumb_tip[1] - 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.circle(frame, tuple(thumb_tip), 15, (0, 255, 255), -1)
                        cv2.circle(frame, tuple(middle_tip), 15, (0, 255, 255), -1)
                    else:
                        cv2.circle(frame, tuple(thumb_tip), 15, (255, 0, 0), -1)
                        cv2.circle(frame, tuple(middle_tip), 15, (255, 0, 0), -1)
                else:
                    if self.left_click_start_time is not None:
                        if self.left_click_held:
                            self.left_click_held = False
                            pyautogui.mouseUp()
                            print("Left Mouse Up (Release)")
                        else:
                            pyautogui.click()
                            print("Left Click Performed")
                    self.left_click_start_time = None

                # Right click detection (thumb + ring finger)
                thumb_ring_dist = self.get_distance(thumb_tip, ring_tip)
                if thumb_ring_dist < self.click_threshold:
                    if not self.right_click_performed:
                        pyautogui.rightClick()
                        self.right_click_performed = True
                        print("Right Click Performed")
                    
                    cv2.circle(frame, tuple(thumb_tip), 15, (0, 0, 255), -1)
                    cv2.circle(frame, tuple(ring_tip), 15, (0, 0, 255), -1)
                    cv2.line(frame, tuple(thumb_tip), tuple(ring_tip), (0, 0, 255), 3)
                    cv2.putText(frame, "RIGHT CLICK", (thumb_tip[0] - 60, thumb_tip[1] - 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.right_click_performed = False
        
        return frame
    
    def run(self):
        """Main function to run the virtual mouse"""
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("Virtual Mouse Control Started!")
        print("Instructions:")
        print("- Point index finger to move mouse")
        print("- Touch thumb and middle finger for a click, hold for drag")
        print("- Touch thumb and ring finger for right click")
        print("- Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = self.process_frame(frame)
            
            cv2.putText(frame, "Enhanced Virtual Mouse Control", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Move in BLUE BOX for full screen control", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Index: Move | T+M: L-Click (Hold to Drag) | T+R: R-Click", 
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Sensitivity: {self.sensitivity}x | Press 'q' to quit", 
                        (10, self.cam_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.namedWindow('Virtual Mouse Control', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Virtual Mouse Control', 640, 360)
            cv2.imshow('Virtual Mouse Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    virtual_mouse = VirtualMouse()
    virtual_mouse.run()
