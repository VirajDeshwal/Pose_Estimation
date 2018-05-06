import cv2
import argparse
import chainer
from pose_detector import PoseDetector, draw_person_pose

chainer.using_config('enable_backprop', False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load model
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)

    cap = cv2.VideoCapture('dj_test.mov')
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # get video frame
        
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if not ret:
            print("Failed to capture image")
            break

        person_pose_array, _ = pose_detector(img)
        data=  draw_person_pose(img, person_pose_array)
        res_img = cv2.addWeighted(img, 0.6, data, 0.4, 0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        
        #cv2.imshow("result", res_img)
        out = cv2.VideoWriter('outpy1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
 
while(True):
  ret, frame = cap.read()
 
  if ret == True: 
     
    # Write the frame into the file 'output.avi'
    out.write(frame)
 
    # Display the resulting frame    
    cv2.imshow('frame',frame)
 
    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else:
    break 
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 

        
        
        
