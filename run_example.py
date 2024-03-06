import cv2
from time import time
import os
import subprocess
import numpy as np
from xycut import bbox2points, recursive_xy_cut, vis_polygons_with_index, recursive_xy_cut_original
from ocr_dla import LayoutLMv3

model = LayoutLMv3("/home/tawat-ki/project/mfec/ocr_dla/tests/data/layoutlmv3-base-finetuned-publaynet/model_final.pth", device="cuda:0")

def inference(img_path,img_out_path):
    layout_object = model(img_path)
    boxes = [box['bbox'] for box in layout_object]
    random_boxes = np.array(boxes)
    index = random_boxes[:,1].argsort()
    random_boxes = random_boxes[index][:-1]

    image = cv2.imread(img_path)
    result = vis_polygons_with_index(image, [bbox2points(it) for it in random_boxes])
    cv2.imwrite(img_out_path.replace(".jpg","_before.jpg"), result)
    # np.random.shuffle(random_boxes)
    res = []
    res_ = []
    print("")
    start_time = time()
    recursive_xy_cut(np.asarray(random_boxes).astype(int), np.arange(len(random_boxes)), res,axis=0)
    print("mine",time()-start_time)
    assert len(res) == len(random_boxes), [len(res),len(random_boxes)]
    sorted_boxes = random_boxes[np.array(res)].tolist()
    # image = cv2.imread(img_path)
    result = vis_polygons_with_index(image, [bbox2points(it) for it in sorted_boxes])
    cv2.imwrite(img_out_path, result)
    # return sorted_boxes

    start_time = time()
    recursive_xy_cut_original(np.asarray(random_boxes).astype(int), np.arange(len(random_boxes)), res_,swap_axis=True)
    print("ori ",time()-start_time)
    sorted_boxes = random_boxes[np.array(res_)].tolist()
    # image = cv2.imread(img_path)
    result = vis_polygons_with_index(image, [bbox2points(it) for it in sorted_boxes])
    cv2.imwrite(img_out_path.replace(".jpg","_ori.jpg"), result)
    return True if res_ == res else False


def main():
    root = "/home/tawat-ki/project/mfec/xy-cut/data"
    stdout = subprocess.run(['ls',root],capture_output=True).stdout.decode("utf-8")
    img_list = [os.path.join(root, v)
                for v in stdout.split("\n")
                if v[-4:] == ".jpg"]
    img_out_list = [os.path.join(root, "results", v)
                    for v in stdout.split("\n")
                    if v[-4:] == ".jpg"]
    results = []
    for i,o in zip(img_list,img_out_list):
        # if i.split("/")[-1] != "test_008.jpg":
        #     continue
        print(f"read:{i}")
        result = inference(i,o)
        results.append(result)
        print("\033[92mpass\033[0m" if result else "\033[93mwrong\033[0m")
        print(f"save:{o}")
        print("-"*80)
    print(f"results: {sum(results)}/{len(results)}")


if __name__ == "__main__":
    main()
