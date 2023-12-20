import cv2
import time
import numpy as np
import openvino as ov
from IPython import display
import matplotlib.pyplot as plt
import sys
# Fetch the notebook utils script from the openvino_notebooks repo
import urllib.request
urllib.request.urlretrieve(
    url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
    filename='notebook_utils.py'
)

sys.path.append("../utils")
import notebook_utils as utils

import subprocess


# directory where model will be downloaded
base_model_dir = "model"

# model name as named in Open Model Zoo
model_name = "face-detection-0205"
# model_name = "facial-landmarks-35-adas-0002"
# model_name = "person-detection-0202"
precision = "FP32"
model_path = (
    f"model/intel/{model_name}/{precision}/{model_name}.xml"
)
download_command = f"omz_downloader " \
                   f"--name {model_name} " \
                   f"--precision {precision} " \
                   f"--output_dir {base_model_dir} " \
                   f"--cache_dir {base_model_dir}"

subprocess.run(download_command, shell=True)

# initialize OpenVINO runtime
core = ov.Core()

# read the network and corresponding weights from file
model = core.read_model(model=model_path)

# compile the model for the CPU (you can choose manually CPU, GPU etc.)
# or let the engine choose the best available device (AUTO)
compiled_model = core.compile_model(model=model, device_name="CPU")

# get input node
input_layer_ir = model.input(0)
N, C, H, W = input_layer_ir.shape
shape = (H, W)

def preprocess(image):
    """
    Define the preprocess function for input data
    
    :param: image: the orignal input frame
    :returns:
            resized_image: the image processed
    """
    resized_image = cv2.resize(image, shape)
    resized_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_BGR2RGB)
    resized_image = resized_image.transpose((2, 0, 1))
    resized_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
    return resized_image


def postprocess(result, image, fps):
    """
    Define the postprocess function for output data
    
    :param: result: the inference results
            image: the orignal input frame
            fps: average throughput calculated for each frame
    :returns:
            image: the image with bounding box and fps message
    """

    
    detections = result.reshape(-1, 5)
    #print(detections)
    
    for i, detection in enumerate(detections):
        xmin, ymin, xmax, ymax, confidence = detection
        if confidence > 0.5:
            
            
            '''
            xmin = int(max((xmin * image.shape[1]), 10))
            ymin = int(max((ymin * image.shape[0]), 10))
            xmax = int(min((xmax * image.shape[1]), image.shape[1] - 10))
            ymax = int(min((ymax * image.shape[0]), image.shape[0] - 10))
            '''
            xmin = int(xmin + 100)
            ymin = int(ymin + 50)
            xmax = int(xmax + 150)
            ymax = int(ymax + 50)

            print(xmin,ymin,xmax,ymax)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.rectangle(image, (xmin, ymin + 30), (xmax, ymax - 110), (0, 0, 0), -1)
            cv2.putText(image, str(round(fps, 2)) + " fps", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3) 
    
    '''
    print(result)
    x0 = int(result[0] * image.shape[1])
    y0 = int(result[1] * image.shape[0])
    x1 = int(result[2] * image.shape[1])
    y1 = int(result[3] * image.shape[0])
    x2 = int(result[4] * image.shape[1])
    y2 = int(result[5] * image.shape[0])
    x3 = int(result[6] * image.shape[1])    
    y3 = int(result[7] * image.shape[0])
    print(image.shape[0],  image.shape[1])
    cv2.line(image, (x0,y0), (x1, y1), (0,0,255), 5)
    cv2.line(image, (x2,y2), (x3, y3), (0,0,255), 5)
    cv2.putText(image, str(round(fps, 2)) + " fps", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3) 
    '''
    '''
    print(result.shape)
    detections = result.reshape(-1, 70)
    detections = detections[0]

    for coor in range(0,70,2) :
        x_coor = int(max((detections[coor] * image.shape[1]), 10))
        y_coor = int(max((detections[coor + 1] * image.shape[0]), 10))
        print(x_coor, y_coor)
        cv2.line(image, (x_coor, y_coor), (x_coor, y_coor), (0, 0, 255), 5)
        cv2.putText(image, str(round(fps, 2)) + " fps", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3) 
    '''
    return image

def sync_api(source=0, flip=False, use_popup=False, skip_first_frames=0):
    """
    Define the main function for video processing in sync mode
    
    :param: source: the video path or the ID of your webcam
    :returns:
            sync_fps: the inference throughput in sync mode
    """
    frame_number = 0
    infer_request = compiled_model.create_infer_request()
    player = None
    try:
        # Create a video player
        player = utils.VideoPlayer(source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # Start capturing
        start_time = time.time()
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        while True:
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            resized_frame = preprocess(frame)
            infer_request.set_tensor(input_layer_ir, ov.Tensor(resized_frame))
            # Start the inference request in synchronous mode 
            infer_request.infer()
            res = infer_request.get_output_tensor(0).data
            #res = res[0][0:8]
            #print(res)
            stop_time = time.time()
            total_time = stop_time - start_time
            frame_number = frame_number + 1
            sync_fps = frame_number / total_time 
            frame = postprocess(res, frame, sync_fps)
            # Display the results
            if use_popup:
                cv2.imshow(title, frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg
                _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                # Create IPython image
                i = display.Image(data=encoded_img)
                # Display the image in this notebook
                display.clear_output(wait=True)
                display.display(i)         
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # Any different error
    except RuntimeError as e:
        print(e)
    finally:
        if use_popup:
            cv2.destroyAllWindows()
        if player is not None:
            # stop capturing
            player.stop()
        return sync_fps
    
sync_fps = sync_api(source=0, flip=False, use_popup=True)
print(f"average throuput in sync mode: {sync_fps:.2f} fps")