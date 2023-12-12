# import libraries
from vidgear.gears import CamGear
import cv2
from ultralytics import YOLO
from torchattacks import PGD, FGSM, FFGSM, VNIFGSM, Pixle, DeepFool
 
# Load the YOLOv8 model
model = YOLO('/home/pigo/projects/adversarial-attacks-pytorch/yolo_adv/best.pt')
model.to(device='cuda', non_blocking=True)
model.training = False
 
# Open the video file
url = "https://www.youtube.com/watch?v=lBlKR2ek0w4"
 
cap = CamGear(source=url, stream_mode=True, logging=True).start() # YouTube Video URL as input

atk = PGD(model=model, yolo=True, eps=0.024, steps=100)
 
# Loop through the video frames
while True:
    # Read a frame from the video
    frame = cap.read()
    
    if frame is None:
        break
    
    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    test = cv2.resize(annotated_frame, (1280, 720))

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", test)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
# Release the video capture object and close the display window
cap.stop()
cv2.destroyAllWindows()