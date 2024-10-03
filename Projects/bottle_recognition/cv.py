import cv2
import numpy as np
from qreader import QReader
import supervision as sv
from ultralytics import YOLOv10
from PIL import Image



nn_model_path = "self_trained_models/07_02_cola_full_empty_v1/weights/last.pt"


def create_qr_dict(image, idx):
    """
    determines the QRs with an NN
    creates a dictionary of the qr codes in right order
    with name as the KEY & their position and center as their VALUE
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a QReader instance
    qreader = QReader()
    # Use the detect_and_decode function to get the decoded QR data
    qrs = qreader.detect_and_decode(image=image, return_detections=True)
    
    
    decoded_info, detections = qrs
    qr_order_box_1 = ["QR1", "QR2", "QR3", "QR4"]
    qr_order_box_2 = ["QR5", "QR6", "QR7", "QR8"]
    qr_centroids = list()
    qr_decoded = tuple()
    
    if idx == 0:
        qr_order = qr_order_box_1
        cam = "CAMERA L"
    else:
        qr_order = qr_order_box_2
        cam = "CAMERA R"
  
    for i in range(len(decoded_info)):
        if decoded_info[i] in qr_order:
            centroid = detections[i]['cxcy']
            qr_centroids.append(centroid)
            qr_decoded += (decoded_info[i],)
    qr_centroids = np.array(qr_centroids).astype(int)
    
    
    qr_dict = dict()
    if set(qr_decoded) != set(qr_order):        
         raise ValueError(f"Robottle couldn't recoginze all of the 4 QR codes in '{cam}': {qr_decoded}. \n   Please retake the image! \n")
    else:
        for qr in qr_order:
            i = qr_decoded.index(qr)
            qr_center = qr_centroids[i]
            qr_dict[qr] = qr_center
        return qr_dict

def get_box_corners(qr_dict):
    """
    extracts the centerpoints of the qr codes from 'qr_dict'
    """
    corners = list()
    for key in qr_dict:
        corners.append(qr_dict[key])
    return corners

def get_birds_eye_view(image, corners):
    """
    transforms the image to the box cut out from top
    """
    # Compute the width and height of the new image, which will be the max distance between points
    widthA = np.linalg.norm(corners[2] - corners[3])
    widthB = np.linalg.norm(corners[1] - corners[0])
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(corners[1] - corners[2])
    heightB = np.linalg.norm(corners[0] - corners[3])
    maxHeight = max(int(heightA), int(heightB))

    # Set the destination points to obtain a "birds eye view"
    pts_dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype='float32')

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, pts_dst)

    # Apply the perspective transformation to the image
    image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return image

def get_niches(image):
    """
    calculates the niches position based on the image frame,
    saved the image as box.jpg,
    returns the niches as an np array
    """
    y_ratio = 15.5 / 27.5 
    x_ratio = 23.5 / 37.5 
    # y_ratio = 17.3 / 27.5 
    # x_ratio = 25.7 / 37.5 
    y_side = image.shape[0] * y_ratio
    x_side = image.shape[1] * x_ratio

    center_y = image.shape[0] / 2 
    center_x = image.shape[1] / 2 


    b1 = np.array([int(center_x - x_side / 2), int(center_y - y_side / 2)])
    b4 = np.array([int(center_x + x_side / 2), int(center_y - y_side / 2)])
    b5 = np.array([int(center_x - x_side / 2), int(center_y)])
    b8 = np.array([int(center_x + x_side / 2), int(center_y)])
    b9 = np.array([int(center_x - x_side / 2), int(center_y + y_side / 2)])
    b12 = np.array([int(center_x + x_side / 2), int(center_y + y_side / 2)])

    def row(p1, p2):
        new_row = np.linspace(p1, p2, 4)
        return new_row.astype(int)

    niches = row(b1, b4)
    niches = np.vstack((niches, row(b5, b8)))
    niches = np.vstack((niches, row(b9, b12)))
    
    

    cv2.imwrite("box.jpg", image)
    return niches

def draw_niches(niches, image):
    """
    marks the niches with an 'x' in the image
    """
    for niche in niches:
        x, y = niche
        cv2.drawMarker(image, (x, y), color=(0, 255, 255), markerType=cv2.MARKER_CROSS, 
                    markerSize=40, thickness=4, line_type=cv2.LINE_AA)
    
    return image
        
def nn_bottles(name, niches, image_fraction):
    """
    a neural net determines the position of full and empty bottles.
    function returns the position of the boundingboxes
    """
    # Load a pretrained YOLOv10n model
    model = YOLOv10(nn_model_path, verbose=False)

    # Perform object detection on an image
    results = model("box.jpg")

    
    image = cv2.imread('./box.jpg')
    image = draw_niches(niches, image)
    detections = sv.Detections.from_ultralytics(results[0])
    # Annotate bounding boxes
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=15)
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)

    # Annotate labels
    label_annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Save the annotated image
    output_path = f'./{name}.jpeg'
    cv2.imwrite(output_path, annotated_image)

    pil_image = Image.open(output_path)
    new_size = (pil_image.width // image_fraction, pil_image.height // image_fraction)
    resized_image = pil_image.resize(new_size)
    resized_image.show()

    return results[0]

def bootle_to_niche(niches, results):
    niches_dict = dict()
    for i in range(niches.shape[0]):
        key = f"niche_{i}"
        value = niches[i]
        niches_dict[key] = [value, 0]

    detections = sv.Detections.from_ultralytics(results)

    for detection in detections:
        # Calculate the center of the bounding box.
        x_min, y_min, x_max, y_max = detection[0]
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        bb_center = np.array([center_x, center_y])

        # Find the point in the list that is closest to the bounding box.
        min_distance = float('inf')
        closest_point = None
        
        for i in range(niches.shape[0]):
            distance = np.linalg.norm(bb_center - niches[i])
            if distance < min_distance:
                min_distance = distance
                closest_niche = i

        if detection[5]['class_name'] == "cola_full":
            niches_dict[f"niche_{closest_niche}"][1] = 2
        elif detection[5]['class_name'] == "cola_empty":
            niches_dict[f"niche_{closest_niche}"][1] = 1

    # print()
    # for niche in niches_dict:
    #     print(niche, ":", niches_dict[niche])
    return niches_dict

def display_niches(key, value):
    """
    displays the bottles box in the terminal
    while X=full_bottle & O=empty_bottle
    """
    print()
    print(key)
    rows = 3
    cols = 4
    
    horizontal_border = "+---" * cols + "+"
    
    for row in range(rows):
        print(horizontal_border)
        for col in range(cols):
            element = value[row * cols + col]
            if element == 2:
                print("| X ", end="")
            elif element == 1:
                print("| O ", end="")
            else:
                print("|   ", end="")
        print("|")
    print(horizontal_border)
  
def run(image_paths, image_fraction):
    """
    returns a dictionary with KEY=camera_name & VALUE=list_of_bottles
    """
    bottle_box_dict = {}
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)

        if idx == 0:
            cam = "CAMERA L"
            name = "camera_l"
        else:
            cam = "CAMERA R"
            name = "camera_r"


        if img is None:
            print("Error: Could not open or find the image.")
        else:
            qr_dict = create_qr_dict(img, idx)
            
            corners = get_box_corners(qr_dict)


            while not (corners[0][0] <= corners[1][0] and corners[0][1] <= corners[3][1] and corners[1][1] <= corners[3][1]):
                print(f"Rotate Image from '{cam}' by 90° ccw")
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                qr_dict = create_qr_dict(img, idx)
                corners = get_box_corners(qr_dict)

            corners = np.array(corners, dtype=np.float32)
            image = get_birds_eye_view(img, corners)
            niches = get_niches(image)
            
            
            nn_results = nn_bottles(name, niches, image_fraction)
            niches_dict = bootle_to_niche(niches, nn_results)

            bottle_box_list = []
            for key in niches_dict.keys():
                bottle_box_list.append(niches_dict[key][1])

            # a dictionary with KEY=camera_name & VALUE=list_of_bottles
            bottle_box_dict[cam] = bottle_box_list
    
    for key, value in bottle_box_dict.items():         
        display_niches(key, value)

    return bottle_box_dict

    


    


