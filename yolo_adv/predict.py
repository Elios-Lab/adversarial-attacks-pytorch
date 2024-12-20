from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2

atk='pixle_new_strong'

model = YOLO('/home/lazzaroni/adv/adversarial-attacks-pytorch/yolo_adv/best.pt')
# image_path = f'/home/pigo/Desktop/test_data/{atk}/32942_Pixle.png'
# results = model(image_path)
results = model.val(data='/home/lazzaroni/adv/adversarial-attacks-pytorch/yolo_adv/data.yaml',split='test', imgsz=1280)
metrics_df = pd.DataFrame.from_dict(results.results_dict, orient='index', columns=['Metric Value'])
print(metrics_df.round(3))


# custom_colors = {
#     0: (0, 255, 0),  # Class 0: Green
#     1: (0, 255, 1),  # Class 1: Yellow
#     2: (255, 0, 0),  # Class 2: Red
#     3: (255,255,255), # Class 3: Off
#     4: (128, 255, 255),  # Class 4: Red-Yellow 
# }
# image = cv2.imread(image_path)

# # Draw bounding boxes on the image
# for result in results:
#     for box in result.boxes:
#         cls = int(box.cls)  # Get the class ID
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extract and convert coordinates to integers
#         color = custom_colors.get(cls, (255, 255, 255))  # Default to white if class not in custom_colors
#         conf = float(box.conf)  # Convert confidence to float
#         if cls==0:
#             cls_name = 'Green'
#         elif cls==1:
#             cls_name = 'Yellow'
#         elif cls==2:
#             cls_name = 'Red'
#         elif cls==3:
#             cls_name = 'Off'
#         elif cls==4:
#             cls_name = 'Red-Yellow'
            
#         label = f"{cls_name} ({conf:.2f})"  # Add class and confidence
        
#         # Draw the bounding box
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        
#         # Add the label
#         label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
#         label_y = max(y1, label_size[1] + 10)
#         cv2.rectangle(image, (x1, label_y - label_size[1] - 5), (x1 + label_size[0], label_y + 5), color, -1)
#         cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)

# # Save the result with bounding boxes
# output_path = f"/home/pigo/Desktop/test_data/{atk}/results/result_32942_Pixle.png"
# cv2.imwrite(output_path, image)

# print(f"Result saved to {output_path}")

    # result.plot(box_color=color)
    # result.show()  # display to screen
    # result.save(filename=f"/home/pigo/Desktop/test_data/{atk}/result.png")  # save to disk