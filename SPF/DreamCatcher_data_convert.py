import soundfile as sf
import numpy as np
from pathlib import Path # 导入 pathlib
import time # 导入 time 模块用于计时

def get_audio_data_correct(file_path):
    """
    读取一个音频文件并返回其数据和元数据。
    在函数内部添加了错误处理，以应对批量任务中可能出现的坏文件。
    """
    try:
        # sf.read() 返回数据和采样率
        audio_data, samplerate = sf.read(file_path)

        # soundfile 返回的形状通常是 (n_frames, n_channels)
        duration = len(audio_data) / samplerate
        channels = audio_data.shape[1] if audio_data.ndim > 1 else 1

        return audio_data, samplerate, channels, duration
    
    except Exception as e:
        # 如果文件损坏或无法读取，打印错误并返回 None
        print(f"  [!] 错误: 无法读取 {file_path}. 错误信息: {e}")
        return None, 0, 0, 0

# --- 批量处理配置 ---

# ！！！请修改这里：指向你包含所有子文件夹的根目录！！！
INPUT_DIRECTORY = Path('.\\test') 

# --------------------------

print(f"--- 🚀 开始批量处理 ---")
print(f"目标根目录: {INPUT_DIRECTORY}")

# 检查输入目录是否存在
if not INPUT_DIRECTORY.is_dir():
    print(f"[!] 错误：目录 {INPUT_DIRECTORY} 不存在。请检查路径。")
else:
    start_time = time.time()
    
    # 1. 使用 rglob 递归查找所有 .wav 文件
    # rglob = Recursive Glob (递归查找)
    wav_files = list(INPUT_DIRECTORY.rglob('*.wav'))
    
    if not wav_files:
        print(f"在 {INPUT_DIRECTORY} 及其子目录中未找到任何 .wav 文件。")
    else:
        print(f"总共找到 {len(wav_files)} 个 .wav 文件。正在开始转换...")
        
        processed_count = 0
        failed_count = 0

        # 2. 遍历找到的每一个 .wav 文件
        for wav_path in wav_files:
            print(f"\n正在处理: {wav_path.name}")
            print(f"  位于: {wav_path.parent}")

            # 3. 生成同名的 .npz 输出路径
            # wav_path.with_suffix('.npz') 会自动将 .wav 替换为 .npz
            output_npz_path = wav_path.with_suffix('.npz')

            # 4. 调用函数读取数据
            audio_data, samplerate, channels, duration = get_audio_data_correct(wav_path)

            # 5. 如果读取失败 (返回了 None)，则跳过此文件
            if audio_data is None:
                failed_count += 1
                continue # 跳到下一个文件

            # 6. 核心代码：保存为 .npz 文件
            try:
                np.savez(
                    output_npz_path,
                    audio_data=audio_data,  # 保存音频数据
                    samplerate=samplerate,  # 保存采样率
                    channels=channels,      # 保存声道数
                    duration=duration       # 保存时长
                )
                print(f"  [✔] 保存成功 -> {output_npz_path.name}")
                processed_count += 1

                # --- (可选) 验证：加载回来检查一下 ---
                print("\n--- 验证加载 ---")
                if processed_count==1:
                    try:
                        loaded_data = np.load(output_npz_path)
                        print(f"加载的文件包含的键: {list(loaded_data.keys())}")
                        print(f"加载的采样率: {loaded_data['samplerate']}")
                        print(f"加载的数据形状: {loaded_data['audio_data'].shape}")
                    except Exception as e:
                        print(f"加载验证失败: {e}")
                
            except Exception as e:
                print(f"  [!] 错误: 无法保存 .npz 文件 {output_npz_path}. 错误信息: {e}")
                failed_count += 1

        # --- 7. 打印最终总结 ---
        end_time = time.time()
        print("\n--- 批量处理完成 ---")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"✅ 成功处理: {processed_count} 个文件")
        print(f"❌ 失败/跳过: {failed_count} 个文件")