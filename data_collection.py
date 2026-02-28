import cv2
import os
import time

def collect_dataset():
    """Capture images from webcam for dataset creation"""
    
    # Create directories if they don't exist
    categories = ['wood_fire', 'liquid_fire', 'electrical_fire']
    for category in categories:
        os.makedirs(f'dataset/{category}', exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("=== FIRE DATASET COLLECTION ===")
    print("Press:")
    print("1 - for Wood/Paper Fire (Class A)")
    print("2 - for Liquid Fire (Class B)")
    print("3 - for Electrical Fire (Class C)")
    print("Q - to quit")
    print("-" * 30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show live feed
        cv2.imshow('Dataset Collection - Point camera at fire source', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('1'):
            # Save as wood fire
            img_name = f"dataset/wood_fire/wood_{int(time.time())}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"✅ Saved as Wood Fire: {img_name}")
            
        elif key == ord('2'):
            # Save as liquid fire
            img_name = f"dataset/liquid_fire/liquid_{int(time.time())}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"✅ Saved as Liquid Fire: {img_name}")
            
        elif key == ord('3'):
            # Save as electrical fire
            img_name = f"dataset/electrical_fire/electrical_{int(time.time())}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"✅ Saved as Electrical Fire: {img_name}")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Dataset collection completed!")

if __name__ == "__main__":
    collect_dataset()