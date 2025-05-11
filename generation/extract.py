import os
import re
import cv2

def parse_filename(filename):
    """解析换脸文件名，提取 targetID, sourceID, videoIndex, frameIndex。"""
    match = re.match(r"(\w+)_(\w+)_(\d+)_(\d+)\.png", filename)
    if match:
        target_id, source_id, video_index, frame_index = match.groups()
        return target_id, source_id, int(video_index), int(frame_index)
    return None, None, None, None

def extract_and_save_frames(txt_file_path, mp4_folder_path, output_folder):
    """
    读取 TXT 文件中的换脸文件名，找到对应的目标视频帧并保存为 PNG。

    Args:
        txt_file_path (str): TXT 文件路径，包含换脸图片文件名。
        mp4_folder_path (str): 存放 MP4 视频文件的文件夹路径。
        output_folder (str): 保存提取帧的输出文件夹名称。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(txt_file_path, 'r') as f:
        for line in f:
            swap_filename = line.strip()
            if not swap_filename:
                continue

            target_id, source_id, video_index, frame_index = parse_filename(swap_filename)

            if target_id is not None and video_index is not None and frame_index is not None:
                target_video_filename = f"{target_id}_{source_id}_{video_index:04d}.mp4"
                video_path = os.path.join(mp4_folder_path, target_video_filename)

                if os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"警告：无法打开视频文件 '{video_path}'")
                        continue

                    # 设置要读取的帧号 (注意 OpenCV 的帧号是从 0 开始的)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()

                    if ret:
                        output_filename = f"{target_id}_{source_id}_{video_index:04d}_{frame_index:05d}.png"
                        output_path = os.path.join(output_folder, output_filename)
                        cv2.imwrite(output_path, frame)
                        print(f"已保存帧: '{output_filename}'")
                    else:
                        print(f"警告：无法从 '{video_path}' 读取第 {frame_index} 帧")

                    cap.release()
                else:
                    print(f"警告：目标视频文件 '{video_path}' 不存在")
            else:
                print(f"警告：无法解析文件名 '{swap_filename}'")

if __name__ == "__main__":
    txt_file = "./image_list.txt"
    mp4_folder = "./Celeb-real"
    output_folder = "./target_frames"
    extract_and_save_frames(txt_file, mp4_folder, output_folder)
    #print("帧提取完成。提取的帧保存在 '{}' 文件夹中。".format("extracted_frames"))
