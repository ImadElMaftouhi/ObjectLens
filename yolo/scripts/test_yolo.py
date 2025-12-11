from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# 1) Load your fine-tuned model
model = YOLO("./yolo/model/weights/best.pt")

# 2) Run inference on one image
# change it
img_path = "./imagenet_yolo15/images/val/n04485082_3738.JPEG" 
result = model(img_path)[0]  # take first image result

# 3) Plot with matplotlib (convert BGR -> RGB)
res_plotted = result.plot()  # this is a NumPy array in BGR
res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

plt.imshow(res_plotted)
plt.axis("off")
plt.show()   # <-- this actually displays the figure
