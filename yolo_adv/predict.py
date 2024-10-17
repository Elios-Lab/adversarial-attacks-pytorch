from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

model = YOLO('yolo_adv/best.pt')

# Run batched inference on a list of images
results = model(["yolo_adv/adv_data/adv/Polter/images/20210616_141500_cam_2_00015734.jpg", "yolo_adv/adv_data/norm/images/20210616_141500_cam_2_00015734.jpg"])  # return a list of Results objects

# Process results list
for i,result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    result.save(filename=f"result_{i}.jpg")  # save to disk







# # results = model.val(data='yolo_adv/data.yaml', imgsz=640)

# results = model("/home/pigo/adversarial-attacks-pytorch/yolo_adv/adv_data/adv/Polter/images/20210616_141500_cam_2_00015734.jpg", verbose=False)
# # print('mAP50-95:', results.box.map)    # map50-95
# # print('mAP50:', results.box.map50)  # map50
# # print('mAP75:', results.box.map75)  # map75
# # print('mAP for each class:', results.box.maps)  # a list contains map50-95 of each category
# # print('Mean precision:', results.box.mp)  # mean precision
# # print('Mean recall:', results.box.mr)

# for r in results:
#     im_array = r.plot()
#     im = Image.fromarray(im_array[..., ::-1])
#     plt.imshow(im)