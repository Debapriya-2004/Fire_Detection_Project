import cv2
import numpy as np
import time

class SimpleFireDetector:
    def __init__(self):
        print("\n" + "="*50)
        print("🔥 FIRE DETECTION & CLASSIFICATION SYSTEM")
        print("="*50)
        print("\nStage 1: Detect fire (any flame)")
        print("Stage 2: Classify fire type\n")

        # Fire type recommendations
        self.recommendations = {
            'wood_fire': {
                'name': 'Wood/Paper Fire (Class A)',
                'agent': '💧 WATER or SAND',
                'warning': 'Safe to use water. Soak thoroughly.',
                'color': (0, 255, 0)  # Green
            },
            'liquid_fire': {
                'name': 'Liquid/Grease Fire (Class B)',
                'agent': '🧯 FOAM EXTINGUISHER',
                'warning': '🚫 NEVER USE WATER! It will spread!',
                'color': (0, 165, 255)  # Orange
            },
            'electrical_fire': {
                'name': 'Gas/Electric Fire (Class C)',
                'agent': '🧯 CO2 or DRY POWDER',
                'warning': '⚡ DO NOT USE WATER! Risk of electrocution!',
                'color': (0, 0, 255)  # Red
            }
        }

        # ---------- STAGE 1: FIRE DETECTION (tighter ranges, higher saturation/value) ----------
        self.detection_ranges = [
            # Orange/Yellow flames (wood, paper) – require good saturation and brightness
            (np.array([0, 100, 150]),  np.array([25, 255, 255])),
            # Blue flames (gas, electric) – narrower blue range, require brightness
            (np.array([100, 80, 150]), np.array([130, 255, 255])),
            # Bright white/yellow (intense fire)
            (np.array([20, 50, 220]),  np.array([40, 255, 255]))
        ]

        # ---------- STAGE 2: FIRE CLASSIFICATION (unchanged) ----------
        self.classification_ranges = {
            'wood_fire':     (np.array([0, 100, 150]), np.array([25, 255, 255])),
            'liquid_fire':   (np.array([20, 100, 180]), np.array([40, 255, 255])),  # bright yellow
            'electrical_fire': (np.array([100, 80, 150]), np.array([130, 255, 255]))
        }

        # Thresholds
        self.detection_min_area = 150        # Increased to avoid tiny spots
        self.min_density = 0.2               # At least 20% of bounding box must be fire-colored
        self.classify_min_area = 200          # Larger region needed for reliable classification
        self.debug_mode = False

        print("✅ System ready!")
        print("\n📸 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 'd' for debug view")
        print("   - Press '+' / '-' to adjust detection sensitivity")
        print("="*50)

    def detect_fire_regions(self, frame):
        """Stage 1: Return only regions that are likely fire."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in self.detection_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        # Clean up mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.detection_min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Compute density: what fraction of the bounding box is actually fire-colored?
            bbox_area = w * h
            density = area / bbox_area
            if density < self.min_density:
                continue   # too sparse – probably not a flame

            # Optional: check average brightness in the region (avoid dark spots)
            roi = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness < 50:   # too dark
                continue

            regions.append((x, y, w, h, area, density))

        return regions, mask

    def classify_fire_region(self, frame, bbox):
        """Stage 2: Given a bounding box, determine fire type."""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return 'wood_fire', 0.0

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        best_type = 'wood_fire'
        best_score = 0.0

        for ftype, (lower, upper) in self.classification_ranges.items():
            mask = cv2.inRange(hsv_roi, lower, upper)
            score = cv2.countNonZero(mask) / (w * h)  # percentage of pixels matching this type
            if score > best_score:
                best_score = score
                best_type = ftype

        # Confidence based on match percentage
        confidence = min(best_score * 1.5, 1.0)
        return best_type, confidence

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("❌ No webcam found!")
            return

        print("✅ Webcam started. Press 'q' to quit.\n")
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # ----- STAGE 1: Detect fire regions -----
            regions, mask = self.detect_fire_regions(frame)

            fire_detected = len(regions) > 0
            if fire_detected:
                # Show detection bounding boxes (yellow) for all regions
                for (x, y, w, h, area, density) in regions:
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 1)

                # ----- STAGE 2: Classify the largest region -----
                largest = max(regions, key=lambda r: r[4])  # by area
                x, y, w, h, area, density = largest

                if area > self.classify_min_area:
                    ftype, conf = self.classify_fire_region(frame, (x, y, w, h))
                else:
                    ftype, conf = 'wood_fire', 0.5  # fallback for small regions

                rec = self.recommendations[ftype]

                # Draw classification bounding box
                cv2.rectangle(display, (x, y), (x+w, y+h), rec['color'], 3)
                label = f"{rec['name']} ({conf*100:.0f}%)"
                cv2.putText(display, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec['color'], 2)

                # Show recommendation panel
                overlay = display.copy()
                cv2.rectangle(overlay, (10, 10), (450, 150), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

                lines = [
                    f"🔥 FIRE DETECTED!",
                    f"Type: {rec['name']}",
                    f"✅ USE: {rec['agent']}",
                    f"⚠️ {rec['warning']}"
                ]
                for i, line in enumerate(lines):
                    color = rec['color'] if i < 2 else (255,255,255)
                    cv2.putText(display, line, (20, 40+i*25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(display, "No fire detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # Debug overlay
            if self.debug_mode and fire_detected:
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask_colored[mask > 0] = (0, 255, 255)  # yellow
                small_mask = cv2.resize(mask_colored, (200,150))
                display[10:160, display.shape[1]-210:display.shape[1]-10] = small_mask

            # Info text
            cv2.putText(display, f"FPS: {fps:.1f}", (10, display.shape[0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(display, f"Regions: {len(regions)} | Sens: {self.detection_min_area}",
                        (10, display.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(display, "Q:Quit D:Debug +/-:Sens", (display.shape[1]-220, display.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            cv2.imshow('Fire Detection & Classification', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                self.detection_min_area += 20
                print(f"Sensitivity decreased (min area = {self.detection_min_area})")
            elif key == ord('-') or key == ord('_'):
                self.detection_min_area = max(50, self.detection_min_area - 20)
                print(f"Sensitivity increased (min area = {self.detection_min_area})")

        cap.release()
        cv2.destroyAllWindows()
        print("✅ System stopped.")

def main():
    print("\n" + "="*50)
    print("🔥 TWO‑STAGE FIRE DETECTION & CLASSIFICATION")
    print("="*50)
    print("\n1. Detects any fire (yellow boxes)")
    print("2. Classifies the largest fire (colored box)")
    input("\nPress ENTER to start...")
    detector = SimpleFireDetector()
    detector.run()

if __name__ == "__main__":
    main()