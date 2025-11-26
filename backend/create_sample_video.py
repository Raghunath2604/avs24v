import cv2
import numpy as np

def create_sample_video(filename='sample_video.mp4', duration=10, fps=30):
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for t in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            frame[y, :, :] = (y // 2, t % 255, 100)
            
        # Moving circle
        cx = int(width/2 + 200 * np.sin(t / 30))
        cy = int(height/2 + 100 * np.cos(t / 30))
        cv2.circle(frame, (cx, cy), 50, (0, 0, 255), -1)
        
        # Text
        cv2.putText(frame, "SentiGuard Sample Feed", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)

    out.release()
    print(f"Created {filename}")

if __name__ == "__main__":
    create_sample_video()
