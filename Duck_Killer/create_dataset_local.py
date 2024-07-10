import numpy as np
import tqdm
import os
import pandas as pd
from PIL import Image
import ast
N_TRAIN_EPISODES = 1
N_VAL_EPISODES = 1



# set path
image_dir_train = '/home/tong/robotic-ai/training_data/episode_0/images'  
csv_file_train = '/home/tong/robotic-ai/training_data/episode_0/data.csv'  

image_dir_val = '/home/tong/robotic-ai/training_data/episode_1/images'  
csv_file_val = '/home/tong/robotic-ai/training_data/episode_1/data.csv'  

def create_real_episode(image_dir, csv_file, path):
    def parse_string_to_list(s):
        return np.asarray(ast.literal_eval(s), dtype=np.float32)
    # 读取CSV文件，这里csv的行列读取是反的，需要注意一下！转换一下在用
    data = pd.read_csv(csv_file, skiprows=1, usecols=range(1, 5), header=None)
    EPISODE_LENGTH = len(data)
    episode = []
    for step in range(EPISODE_LENGTH):
        # 读取图片
        image_path = os.path.join(image_dir, f'image_{step}.png')  
        
        image = np.array(Image.open(image_path), dtype=np.uint8)
        
        # 将 wrist_image 设置为空
        wrist_image = np.zeros((64, 64, 3), dtype=np.uint8)

        # 获取当前行的数据并解析
        row_data = data.iloc[step]
        # 将列转换为列表并解析
        state1 = parse_string_to_list(row_data[1]) 
        state2 = parse_string_to_list(row_data[2])
        state = np.concatenate((state1, state2))
        action = parse_string_to_list(row_data[4])  
        episode.append({
            'image': image,
            'wrist_image': wrist_image,
            'state': state,
            'action': action,
            'language_instruction': 'duck killer',  # 固定的语言指令
        }) 
    np.save(path, episode)


# create real episodes for train and validation
print("Generating train examples...")
os.makedirs('./data/train', exist_ok=True)
for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
    create_real_episode(image_dir_train, csv_file_train, f'./data/train/episode_{i}.npy')

print("Generating val examples...")
os.makedirs('./data/val', exist_ok=True)
for i in tqdm.tqdm(range(N_VAL_EPISODES)):
    create_real_episode(image_dir_val, csv_file_val,f'./data/val/episode_{i}.npy')

print('Successfully created example data!')
