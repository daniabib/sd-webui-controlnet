import pyelsed
import cv2
import numpy as np

from pathlib import Path


"""
pyelsed.detect(img: numpy.ndarray, sigma: float = 1, gradientThreshold: float = 30, minLineLen: int = 15, lineFitErrThreshold: float = 0.2, pxToSegmentDistTh: float = 1.5, validationTh: float = 0.15, validate: bool = True, treatJunctions: bool = True) -> tuple
"""

# def apply_elsed(input_image, gradientThreshold=65.0, validationTh=0.5):
def apply_elsed(input_image, thr_a=65.0, thr_b=0.5):
    
    try:
        img = cv2.imread(str(input_image), cv2.IMREAD_GRAYSCALE)
        segments, scores = pyelsed.detect(img, 
                                        gradientThreshold=thr_a,
                                        validationTh=thr_b)

        ## Saving image similar to MLSD CNet webui extension
        img_output = np.zeros_like(img)

        dbg = cv2.cvtColor(img_output, cv2.COLOR_GRAY2RGB)
        for s in segments.astype(np.int32):
            cv2.line(dbg, (s[0], s[1]), (s[2], s[3]), (255, 255, 255), 1, cv2.LINE_AA)
        return dbg
        
    except Exception as e:
        print(e)


# count = 564
# for i in range(count, len(images)):
#     if images[i].suffix.lower() in file_extensions:
#         image_number = images[i].stem.split("_")[-1]
#         if int(image_number) not in bad_images:
#             try:
#                 ic(f"Image: {count}, {image_number}")
#                 img = cv2.imread(str(images[i]), cv2.IMREAD_GRAYSCALE)
#                 segments, scores = pyelsed.detect(img, 
#                                                 gradientThreshold=65.0,
#                                                 validationTh=0.5)

#                 ## Saving image similar to MLSD CNet webui extension
#                 img_output = np.zeros_like(img)

#                 dbg = cv2.cvtColor(img_output, cv2.COLOR_GRAY2RGB)
#                 for s in segments.astype(np.int32):
#                     cv2.line(dbg, (s[0], s[1]), (s[2], s[3]), (255, 255, 255), 1, cv2.LINE_AA)

#                 cv2.imwrite(f"{ouput_path}/archviz_elsed_1024_{image_number:07}.png", dbg)

#             except Exception as e:
#                 # Log the error to a file
#                 with open("bad_images_elsed.txt", "a") as file:
#                     file.write(f"Error with image {image_number}: {e}\n")
#     count += 1