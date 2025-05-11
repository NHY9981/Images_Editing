import os
import argparse

def read_existing_videos(txt_file):
    """读取已有的测试视频文件列表"""
    existing_videos = set()
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # 提取视频路径部分
                video_path = parts[1]
                existing_videos.add(video_path)
    return existing_videos

def find_new_videos(base_dir, existing_videos):
    """查找新的视频并标记标签"""
    new_videos = []
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # 检查文件是否为视频文件
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # 获取相对路径
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                
                # 如果该视频不在已有列表中
                if rel_path not in existing_videos:
                    # 判断标签：如果文件夹名称中包含"real"则为1，否则为0
                    label = 1 if "real" in os.path.dirname(rel_path).lower() else 0
                    new_videos.append((label, rel_path))
    
    return new_videos

def save_video_list(videos, output_file):
    """保存视频列表到文件"""
    with open(output_file, 'w') as f:
        for label, path in videos:
            f.write(f"{label} {path}\n")

def main():
    parser = argparse.ArgumentParser(description="处理视频文件并生成标签列表")
    parser.add_argument("--input_dir", type=str, default="/root/autodl-tmp/Celeb-DF-v2", help="视频文件所在的根目录")
    parser.add_argument("--existing_list", type=str, default="/root/autodl-tmp/Celeb-DF-v2/List_of_testing_videos.txt", help="已有的视频列表文件")
    parser.add_argument("--output_file", type=str, default="/root/our_model/train-list.txt", help="输出的新视频列表文件")
    
    args = parser.parse_args()
    
    # 读取已有视频列表
    existing_videos = read_existing_videos(args.existing_list)
    print(f"已有视频列表中有 {len(existing_videos)} 个视频文件")
    
    # 查找新视频
    new_videos = find_new_videos(args.input_dir, existing_videos)
    print(f"找到 {len(new_videos)} 个新视频文件")
    
    # 保存结果
    save_video_list(new_videos, args.output_file)
    print(f"结果已保存到 {args.output_file}")

if __name__ == "__main__":
    main()