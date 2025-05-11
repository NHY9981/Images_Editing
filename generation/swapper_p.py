"""
This project is developed by Haofan Wang to support face swap in single frame. Multi-frame will be supported soon!

It is highly built on the top of insightface, sd-webui-roop and CodeFormer.
"""

import os
import cv2
import copy
import argparse
import insightface
import onnxruntime
import numpy as np
from PIL import Image
import random
from typing import List, Union, Dict, Set, Tuple


def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def getFaceAnalyser(model_path: str, providers,
                    det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser,
                 frame:np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

    
def get_many_faces(face_analyser,
                   frame:np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(face_swapper,
              source_faces,
              target_faces,
              source_index,
              target_index,
              temp_frame):
    """
    paste source_face on target image
    """
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]

    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
 
    
def process(source_img: Union[Image.Image, List],
            target_img: Image.Image,
            source_indexes: str,
            target_indexes: str,
            model: str):
    # load machine default available providers
    #providers = ['CUDAExecutionProvider']
    providers = onnxruntime.get_available_providers()
    #print(f"Available providers: {providers}")

    # load face_analyser
    face_analyser = getFaceAnalyser(model, providers)
    
    # load face_swapper
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = getFaceSwapModel(model_path)
    
    # read target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    # detect faces that will be replaced in the target image
    target_faces = get_many_faces(face_analyser, target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        if isinstance(source_img, list) and num_source_images == num_target_faces:
            print("Replacing faces in target image from the left to the right by order")
            for i in range(num_target_faces):
                source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                source_index = i
                target_index = i

                if source_faces is None:
                    raise Exception("No source faces found!")

                temp_frame = swap_face(
                    face_swapper,
                    source_faces,
                    target_faces,
                    source_index,
                    target_index,
                    temp_frame
                )
        elif num_source_images == 1:
            # detect source faces that will be replaced into the target image
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
            num_source_faces = len(source_faces)
            print(f"Source faces: {num_source_faces}")
            print(f"Target faces: {num_target_faces}")

            if source_faces is None:
                raise Exception("No source faces found!")

            if target_indexes == "-1":
                if num_source_faces == 1:
                    print("Replacing all faces in target image with the same face from the source image")
                    num_iterations = num_target_faces
                elif num_source_faces < num_target_faces:
                    print("There are less faces in the source image than the target image, replacing as many as we can")
                    num_iterations = num_source_faces
                elif num_target_faces < num_source_faces:
                    print("There are less faces in the target image than the source image, replacing as many as we can")
                    num_iterations = num_target_faces
                else:
                    print("Replacing all faces in the target image with the faces from the source image")
                    num_iterations = num_target_faces

                for i in range(num_iterations):
                    source_index = 0 if num_source_faces == 1 else i
                    target_index = i

                    temp_frame = swap_face(
                        face_swapper,
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame
                    )
            else:
                print("Replacing specific face(s) in the target image with specific face(s) from the source image")

                if source_indexes == "-1":
                    source_indexes = ','.join(map(lambda x: str(x), range(num_source_faces)))

                if target_indexes == "-1":
                    target_indexes = ','.join(map(lambda x: str(x), range(num_target_faces)))

                source_indexes = source_indexes.split(',')
                target_indexes = target_indexes.split(',')
                num_source_faces_to_swap = len(source_indexes)
                num_target_faces_to_swap = len(target_indexes)

                if num_source_faces_to_swap > num_source_faces:
                    raise Exception("Number of source indexes is greater than the number of faces in the source image")

                if num_target_faces_to_swap > num_target_faces:
                    raise Exception("Number of target indexes is greater than the number of faces in the target image")

                if num_source_faces_to_swap > num_target_faces_to_swap:
                    num_iterations = num_source_faces_to_swap
                else:
                    num_iterations = num_target_faces_to_swap

                if num_source_faces_to_swap == num_target_faces_to_swap:
                    for index in range(num_iterations):
                        source_index = int(source_indexes[index])
                        target_index = int(target_indexes[index])

                        if source_index > num_source_faces-1:
                            raise ValueError(f"Source index {source_index} is higher than the number of faces in the source image")

                        if target_index > num_target_faces-1:
                            raise ValueError(f"Target index {target_index} is higher than the number of faces in the target image")

                        temp_frame = swap_face(
                            face_swapper,
                            source_faces,
                            target_faces,
                            source_index,
                            target_index,
                            temp_frame
                        )
        else:
            raise Exception("Unsupported face configuration")
        result = temp_frame
    else:
        print("No target faces found!")
    
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image


def parse_args():
    parser = argparse.ArgumentParser(description="Face swap.")
    parser.add_argument("--source_img_folder", type=str, required=True, help="The path of source image, it can be multiple images, dir;dir2;dir3.")
    parser.add_argument("--target_img_folder", type=str, required=True, help="The path of target image.")
    parser.add_argument("--output_img_folder", type=str, required=True, help="The path and filename of output image.")
    parser.add_argument("--source_indexes", type=str, required=False, default="-1", help="Comma separated list of the face indexes to use (left to right) in the source image, starting at 0 (-1 uses all faces in the source image")
    parser.add_argument("--target_indexes", type=str, required=False, default="-1", help="Comma separated list of the face indexes to swap (left to right) in the target image, starting at 0 (-1 swaps all faces in the target image")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    
    source_frames_dir = args.source_img_folder
    
    target_frames_dir = args.target_img_folder

    output_frames_dir = args.output_img_folder
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    image_list_path = "./image_list.txt"

    model = "./inswapper_128.onnx"


    try:
        with open(image_list_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：找不到文件 {image_list_path}")
        exit(1)
    
    source_images = {}
    for filename in os.listdir(source_frames_dir):
        parts = filename.split('_')
        if len(parts) > 0:
            source_id = parts[0]
            if source_id not in source_images:
                source_images[source_id] = []
            source_images[source_id].append(os.path.join(source_frames_dir, filename))

    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split('_')
        if len(parts) == 4:
            target_id = parts[0]
            source_id = parts[1]
            video_index = parts[2]
            frame_index = parts[3]

            target_filename = f"{target_id}_{video_index}_{frame_index}"
            target_path = os.path.join(target_frames_dir, target_filename)

            if os.path.exists(target_path):
                print(f"找到目标图片: {target_path}")

                if source_id in source_images and source_images[source_id]:
                    random_source_path = random.choice(source_images[source_id])
                    print(f"随机选择源图片: {random_source_path}")
            source_img = [Image.open(random_source_path)]
            target_img = Image.open(target_path)

            result_image = process(source_img, target_img, args.source_indexes, args.target_indexes, model)

            # save result
            result_filename = f"{target_id}_{source_id}_{video_index}_{frame_index}"
            result_path = os.path.join(output_frames_dir, result_filename)
            result_image.save(result_path)
            print(f"结果保存成功: {result_path}")
            
            # Increment and display the count of processed images
            if not hasattr(process, "processed_count"):
                process.processed_count = 0
            process.processed_count += 1
            print(f"已处理图片数量: {process.processed_count}")


    
    
    
    
