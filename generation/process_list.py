import os
import random
import cv2

def process_image_list(image_list_path, video_root_dir, output_dir):
    """
    读取 image_list.txt，找到对应的视频文件（以 target_id 开头），
    在每个 target_id 的视频中随机选取一帧保存到指定文件夹，并进行去重。

    Args:
        image_list_path (str): image_list.txt 文件的路径。
        video_root_dir (str): 包含所有视频文件的根目录。
        output_dir (str): 保存随机选取帧的输出文件夹路径。
    """
    try:
        with open(image_list_path, 'r') as f:
            image_lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：找不到文件 {image_list_path}")
        return

    os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹，如果不存在
    processed_ids = set()  # 用于存储已处理过的 target_id

    for line in image_lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split('_')
        if len(parts) >= 2:
            target_id = parts[1]

            if target_id in processed_ids:
                print(f"跳过 target_id: {target_id}，已处理过。")
                continue

            found_videos = []
            for filename in os.listdir(video_root_dir):
                if filename.startswith(target_id) and filename.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(video_root_dir, filename)
                    found_videos.append(video_path)

            for video_index, selected_video_path in enumerate(found_videos):
                cap = cv2.VideoCapture(selected_video_path)
                if not cap.isOpened():
                    print(f"警告：无法打开视频文件 {selected_video_path}")
                    continue

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count > 0:
                    random_frame_number = random.randint(0, frame_count - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
                    ret, frame = cap.read()
                    if ret:
                        video_name_without_ext = os.path.splitext(os.path.basename(selected_video_path))[0]
                        output_filename = f"{video_name_without_ext}_{random_frame_number:05d}.png"
                        output_path = os.path.join(output_dir, output_filename)
                        cv2.imwrite(output_path, frame)
                        print(f"已保存帧: {output_filename} 来自视频: {selected_video_path}")
                    else:
                        print(f"警告：无法读取视频 {selected_video_path} 的第 {random_frame_number} 帧。")
                else:
                    print(f"警告：视频 {selected_video_path} 没有帧。")
                cap.release()

            processed_ids.add(target_id)  # 将处理过的 target_id 添加到集合中
        else:
            print(f"警告：无法解析行 '{line}'，跳过。")


if __name__ == "__main__":
    image_list_file = 'image_list.txt'
    video_directory = './Celeb-real'
    output_directory = './source_frames'  # 指定输出文件夹

    process_image_list(image_list_file, video_directory, output_directory)
    print("处理完成。")
