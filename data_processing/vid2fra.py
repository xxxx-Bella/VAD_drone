import os
import cv2


# 设置视频文件路径和帧保存目录
video_dir = '../Campus/playground/videos/abnormal/'
frame_dir = '../Campus/playground/frames/abnormal/'

# 遍历视频文件
for filename in os.listdir(video_dir):
    if filename.endswith(".mp4") or filename.endswith(".MP4"):
        # 获取视频文件名（不含扩展名）
        video_name = os.path.splitext(filename)[0]
        print('video_name:', video_name)
        # 创建对应帧保存目录
        frame_path = os.path.join(frame_dir, video_name)
        os.makedirs(frame_path, exist_ok=True)

        # 打开视频文件
        cap = cv2.VideoCapture(os.path.join(video_dir, filename))
        # 逐帧读取视频并保存为图像文件
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dir = os.path.join(frame_path, f"{count}.jpg")
            success = cv2.imwrite(dir, frame)
            count += 1
            # if success:
            #     print('transformation done!')
            # else:
            #     print('transformation unsuccessful!')
        # 关闭视频文件
        cap.release()

print('transformation done!')
