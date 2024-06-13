import cv2

def detect_leaves(image):
    # Load pre-trained object detection model
    net = cv2.dnn.readNetFromDarknet('path/to/config_file.cfg', 'path/to/weights_file.weights')

    # Specify the classes to detect (in this case, only leaves)
    classes = ['leaf']

    # Get the output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Read the image
    img = cv2.imread(image)

    # Resize the image to a standard size
    img = cv2.resize(img, (416, 416))

    # Convert the image to a blob
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)

    # Set the input for the neural network
    net.setInput(blob)

    # Perform forward pass and get the output
    outputs = net.forward(output_layers)

    # Initialize lists for bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each output layer
    for output in outputs:
        # Loop over each detection
        for detection in output:
            # Get the confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections
            if confidence > 0.5:
                # Get the center coordinates and dimensions of the bounding box
                center_x = int(detection[0] * img.shape[1])
                center_y = int(detection[1] * img.shape[0])
                width = int(detection[2] * img.shape[1])
                height = int(detection[3] * img.shape[0])

                # Calculate the top-left corner coordinates of the bounding box
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Add the bounding box, confidence, and class ID to the respective lists
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    for i in indices:
        i = i[0]
        x, y, width, height = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]

        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(img, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image with bounding boxes
    cv2.imshow('Leaf Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = '/home/josh/Downloads/download.jpeg'
detect_leaves(image_path)