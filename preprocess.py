import skimage.io as io
import numpy as np
from skimage import img_as_float, img_as_ubyte, exposure
from skimage.filters import median
from scipy import ndimage
from skimage.morphology import disk, binary_closing, remove_small_holes
import os
import cv2
from tqdm import tqdm

threshold_config_dict = {
    '1': 1.8,
    '2': 1.2,
    '3': 1.5,
    '4': 1.5,
    '5': 1.5,
    '6': 1.7,
    '7': 0.7,
    '8': 0.8,
    '9': 1.4,
    '10': 1.2
}
median_config_dict = {
    '1': 3,
    '2': 5,
    '3': 7,
    '4': 7.5,
    '5': 7,
    '6': 7,
    '7': 5,
    '8': 4,
    '9': 5.7,
    '10': 4
}

def process_image(path, threshold_config,median_config, method):
    img = io.imread(path)
    img = img_as_float(img)

    if len(img.shape) == 3:
        img_gray = np.mean(img, axis=2)
    else:
        img_gray = img.copy()
    filtered_img = median(img_gray, disk(median_config))

    img2 = (filtered_img - filtered_img.min()) / (filtered_img.max() - filtered_img.min())
    threshold = img2.mean() - img2.std() * threshold_config
    img2[img2 < threshold] = 0
    img2[img2 >= threshold] = 1

    if method == 5:
        img2 = img2.astype(bool)
        h, w = img2.shape
        visited = np.zeros((h, w), dtype=bool)
        
        filled_img = img2.copy()
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        from collections import deque
        queue = deque()

        for x in range(w):
            if img2[0, x]:
                queue.append((0, x))
                visited[0, x] = True
            if img2[h-1, x]:
                queue.append((h-1, x))
                visited[h-1, x] = True
        for y in range(h):
            if img2[y, 0]:
                queue.append((y, 0))
                visited[y, 0] = True
            if img2[y, w-1]:
                queue.append((y, w-1))
                visited[y, w-1] = True

        while queue:
            y, x = queue.popleft()
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if img2[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        filled_img[~visited] = False

        img2 = filled_img
        

    elif method == 2:
        img2 = img2.astype(bool)
        img2 = remove_small_holes(img2, area_threshold=50)
    elif  method == 3:
        img2= binary_closing(img2, disk(1))
    elif method == 4:
        img2_uint8 = img2.astype(np.uint8) * 255  
        kernel = np.ones((3, 3), np.uint8) 
        img2_closed = cv2.morphologyEx(img2_uint8, cv2.MORPH_CLOSE, kernel)  

        contours, _ = cv2.findContours(img2_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 1000:  
                cv2.drawContours(img2_closed, [contour], -1, 255, thickness=cv2.FILLED)

        img2 = img2_closed.astype(float) / 255.0  
    img2 = img2.astype(float)
    return img, img2

def process_test_dataset():
    dataset_path = "test"
    font = cv2.FONT_HERSHEY_SIMPLEX  

    for i in os.listdir(dataset_path):
        if not os.path.exists(os.path.join(dataset_path, f"res_{i}")):
            os.makedirs(os.path.join(dataset_path, f"res_{i}"), exist_ok=True)

        if os.path.isdir(os.path.join(dataset_path, i)):
            continue
        if i.endswith(".jpg"):
            p = os.path.join(dataset_path, i)
            threshold_config = threshold_config_dict[i[0]]
            median_config = median_config_dict[i[0]]
            output_dir = os.path.join(dataset_path, f"res_{i}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            method_results = []
            method_names = ["normalization", "remove_small_hole", "binary_closing", "cv2", "bfs"]
            
            for method in range(1, 6, 1):
                img, img2 = process_image(p, threshold_config,median_config, method)

                img_combined = (img + 1) / 2
                img_combined = np.concatenate((img_combined, img2), axis=1)

                img_combined = exposure.rescale_intensity(img_combined, out_range=(0, 1))
                img_combined = img_as_ubyte(img_combined)

                method_name = method_names[method - 1]
                img_combined = cv2.putText(img_combined, method_name, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

                method_results.append(img_combined)

                output_path = os.path.join(output_dir, f"{os.path.splitext(i)[0]}_method_{method_name}.jpg")
                io.imsave(output_path, img_combined)

            final_combined = np.vstack(method_results)

            final_output_path = os.path.join(output_dir, f"{os.path.splitext(i)[0]}_combined_results.jpg")
            io.imsave(final_output_path, final_combined)

        print(f"Done: {i}")


def process_full_dataset():
    # dataset_path = "dataset"
    dataset_path = "/data1/raytsai/touching_"
    data_path = "/data1/raytsai/test"
    res_path = "/data1/raytsai/test_mask"
    os.makedirs(data_path, exist_ok=True) 
    os.makedirs(res_path, exist_ok=True) 
    for i in os.listdir(dataset_path):
        t = 0
        # res_path = os.path.join(dataset_path, f"res_{i}")
        # os.makedirs(res_path, exist_ok=True) 

        if not i[0].isdigit():
            continue
        # 讀取子資料夾中的 .tif 檔案
        tif_files = [j for j in os.listdir(os.path.join(dataset_path, i)) if j.endswith(".tif")]
        # if i != '9':
        #     continue
        for j in tqdm(tif_files, desc=f"Processing files in folder {i}"):
            # if j[4] == '.' or j[3] == '.' :
            #     continue

            # if  int(j[2:5]) > 902 or int(j[2:5]) < 902:
            #     continue
            t += 1
            # if t == 21:
            #     break
            p = os.path.join(dataset_path, i, j)
            
            threshold_config = threshold_config_dict.get(i[0])  
            median_config = median_config_dict.get(i[0])
            img, img2 = process_image(p, threshold_config,median_config, 5)

            # save fig
            # img_combined = (img + 1) / 2
            # img_combined = np.concatenate((img_combined, img2), axis=1)
            img =  exposure.rescale_intensity(img, out_range=(0, 1))
            img2 =  exposure.rescale_intensity(img2, out_range=(0, 1))
            img = img_as_ubyte(img)
            img2 = img_as_ubyte(img2)
            # img_combined = exposure.rescale_intensity(img_combined, out_range=(0, 1))
            # img_combined = img_as_ubyte(img_combined)
            output_path = os.path.join(data_path, f"{os.path.splitext(j)[0]}.jpg")
            output_mask_path = os.path.join(res_path, f"{os.path.splitext(j)[0]}_mask.jpg")
            io.imsave(output_path, img)
            io.imsave(output_mask_path, img2)
            
        print(f"Done: {i}")
    

def main():

    # test==1 for select preprocess method
    # test==0 for preprocess data with bfs method
    testing = 0
    
    if testing == 1:
        os.system('cd test; rm -rf res_*; cd ..')
        process_test_dataset()
    else:
        # os.system('cd dataset; rm -rf res_*; cd ..')
        process_full_dataset()

if __name__ == "__main__":
    main()