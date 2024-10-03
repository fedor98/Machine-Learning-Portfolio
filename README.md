Hey there,

You are looking at some sort of lil representation of my current progress in the field of machine learning. When I'm done whith certein part or find time to fully train the model (as it seems my machine isn't quite suitable for the high computational task) I usually try to upload the results at some form. I'm just at the beginning of my journey, so this is more of a work in progress than something final. Feel free to take a look.

---

## Projects:

### Recognition of full vs. empty Coke bottles [[Link to short video demo](Projects/bottle_recognition/results/bottle_recognition.mp4), [Link to .py file](Projects/bottle_recognition/cv.py)]

- recognition of Coke bottles and assignment of these bottles to niches in a box with the help of 4 qr codes
  - provision of information to a robotic arm [[Link to video of the sorting roboter](Projects/bottle_recognition/results/robotic_arm_in_action.mp4)]
- Tags (OpenCV, YOLOv10, fine-tuning of the coco-model, qr code detection, creation and annotation of an image dataset)
- _if you want to recreate the inference process use the [README.md](Projects/bottle_recognition/README.md) after you have downloaded the project folder and watched the [video demo](Projects/bottle_recognition/results/bottle_recognition.mp4)_
  - _you will also find some example photos in the `exmaple_photos` folder, which you have to copy in the `photos` folder_
    <img width="500" alt="bottle_recognition" src="">

---

### License Plate Tracker [[Link to the .ipynb file](Projects/license_plate_detector/license_plate.ipynb)]

- Tags (OpenCV, YOLOv8, fine-tuning of the coco-model, ocr of license plates)
- classification of cancerous vs. non-cancerous images
- Tags (OpenCV, YOLOv8, fine-tuning of the coco-model, ocr of license plates)
- _the following gives a good baseline licence plate tracking model, but it needs some improvemetnt:_
  - _transfer learning on a larger license plate dataset for better object detection_
  - _using a custom OCR model which was trained on regional license plates as easyocr struggles with the continuous classification of characters in this sample footage_
    <img width="500" alt="license_plate_tracking" src="https://github.com/fedor98/Machine-Learning-Portfolio/assets/136340206/06b7bc5f-6f27-478c-b8c6-dd0bb42cc63c">

---

### Google & Apple stock prediction [[Link to the .ipynb file](Projects/stock_prediction/stock_prediction_9days.ipynb)]

- prediction of the future stock development based on the previous closing prizes
- Tags (Tensorflow, RNN)
  <img width="500" alt="stock_prediction_9days" src="https://github.com/fedor98/Machine-Learning-Portfolio/assets/136340206/d8b655a8-3585-4d4b-9735-e14e41b03604">

---

### Breast Cancer [[Link to the .ipynb file](Projects/breast_histopathology/breast_histopathology_shortened.ipynb)]

- classification of cancerous vs. non-cancerous images
- Tags (Tensorflow, oversampling, transfer learning, unfreezing the base model)
  <img width="500" alt="breast_histopathology" src="https://github.com/fedor98/Machine-Learning-Portfolio/assets/136340206/24737a39-9795-4ab1-99f9-edc084092089">

---

### CIFAR 100 [[Link to the .ipynb file](Projects/Cifar100/cifar100_shortened.ipynb)]

- Image classification of 100 classes
- Tags (Tensorflow, data augmentation)
  <img width="500" alt="cifar100" src="https://github.com/fedor98/Machine-Learning-Portfolio/assets/136340206/6e5230c7-bdef-4f96-873e-141125e743a9">

---

### Cats vs. Dogs [[Link to the .ipynb file](Projects/CatsVsDogs/cats_vs_dogs.ipynb)]

- Image classification of 2 classes
- Tags (Tensorflow, transfer learning)
  <img width="500" alt="cats_vs_dogs" src="https://github.com/fedor98/Machine-Learning-Portfolio/assets/136340206/e5026c9f-f63b-4ef9-94bc-37ab8fc436c5">

---

### Comparison of Activation Function carried out on MNIST [[Link to the .ipynb file](Projects/mnist/mnist_prediction_of_numbers.ipynb)]

- Image classification of 2 classes
- Tags (Tensorflow, relu, swish, mish, selu, gelu)
  <img width="500" alt="mnist" src=https://github.com/fedor98/Machine-Learning-Portfolio/assets/136340206/f9ee6e22-f0e1-4255-9acf-96fb83b0becc)>
