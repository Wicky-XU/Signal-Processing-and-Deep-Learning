#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Transcript Project - 主程序
自动视频转录系统主入口

功能:
1. 人脸识别 - 识别视频中的说话者
2. 语音识别 - 将语音转换为文本
3. 自动翻译 - 将法语翻译为英语
4. 时间戳标记 - 为每句话添加时间戳
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

# 导入项目模块
try:
    from src.face_recognition import FaceRecognizer
    from src.speech_recognition import SpeechRecognizer
    from src.translation import Translator
    from src.utils import (
        check_dependencies,
        validate_input_file,
        save_transcript_results,
        print_project_banner,
        cleanup_temp_files,
        check_models_cache,
        setup_models_cache,
        print_models_info
    )
    
    MODULES_OK = True
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保src目录下的所有模块文件存在")
    print("如果是人脸识别模块失败，请安装: pip install facenet-pytorch torch torchvision")
    MODULES_OK = False


def auto_transcript(input_file, output_file):
    """
    主转录函数
    
    Args:
        input_file: 输入视频文件路径
        output_file: 输出转录文件路径
        
    Returns:
        bool: 转录是否成功
    """
    
    if not MODULES_OK:
        print("模块未正确加载，无法执行转录")
        return False
    
    print(f"开始处理视频: {input_file}")
    
    try:
        # Step 1: 人脸识别
        print("\nStep 1: 人脸识别")
        print("-" * 30)
        
        try:
            face_recognizer = FaceRecognizer()
            speaker_name = face_recognizer.identify_speaker(input_file)
            print(f"人脸识别完成，说话者: {speaker_name}")
        except Exception as e:
            print(f"人脸识别出现错误: {e}")
            print("使用默认说话者名称")
            speaker_name = "Speaker"
        
        # Step 2: 语音识别
        print(f"\nStep 2: 语音识别")
        print("-" * 30)
        speech_recognizer = SpeechRecognizer()
        transcript_data = speech_recognizer.transcribe_video(input_file)
        
        if not transcript_data:
            print("未能识别出任何语音内容")
            return False
        
        # Step 3: 翻译处理
        print(f"\nStep 3: 翻译处理")
        print("-" * 30)
        translator = Translator()
        final_results = translator.process_transcript(transcript_data, speaker_name)
        
        # 显示结果
        print(f"\n转录结果:")
        print("=" * 60)
        for line in final_results:
            print(line)
        print("=" * 60)
        
        # 保存结果
        success = save_transcript_results(final_results, output_file, input_file)
        
        if success:
            print(f"转录完成! 结果已保存到: {output_file}")
            return True
        else:
            print("保存结果时出现错误")
            return False
        
    except Exception as e:
        print(f"转录过程中出现错误: {e}")
        return False
    
    finally:
        # 清理临时文件和GPU缓存
        cleanup_temp_files()
        
        # 清理GPU缓存（如果使用了GPU）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def find_input_video():
    """
    查找输入视频文件
    
    Returns:
        str: 视频文件路径，如果未找到返回None
    """
    input_dir = Path("data/input")
    
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        print("请创建data/input目录并放置视频文件")
        return None
    
    # 查找视频文件
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(str(input_dir / ext)))
    
    if not video_files:
        print(f"在{input_dir}目录下未找到视频文件")
        print(f"支持的格式: {', '.join(video_extensions)}")
        return None
    
    # 返回第一个找到的视频文件
    return video_files[0]


def check_face_recognition_dependencies():
    """检查人脸识别专用依赖"""
    try:
        import torch
        import torchvision
        from facenet_pytorch import MTCNN, InceptionResnetV1
        
        print("深度学习依赖检查通过")
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU不可用，将使用CPU（速度较慢）")
        
        return True
        
    except ImportError as e:
        print(f"人脸识别依赖缺失: {e}")
        print("请安装: pip install torch torchvision facenet-pytorch")
        return False
    """
    生成输出文件名
    
    Returns:
        str: 输出文件路径
    """
def generate_output_filename():
    """
    生成输出文件名
    
    Returns:
        str: 输出文件路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path("data/output")
    return str(output_dir / f"transcript_{timestamp}.txt")


def main():
    """主函数"""
    
    # 打印项目信息
    print_project_banner()
    
    # 检查基础依赖
    if not check_dependencies():
        print("\n请安装缺失的基础依赖包后重新运行程序")
        return
    
    # 检查人脸识别专用依赖
    if not check_face_recognition_dependencies():
        print("\n人脸识别功能将受限，建议安装深度学习依赖包")
        
        # 询问是否继续
        try:
            user_input = input("是否继续运行（可能影响人脸识别准确性）？(y/n): ")
            if user_input.lower() not in ['y', 'yes', '是']:
                print("程序退出")
                return
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            return
    else:
        # 检查和设置模型缓存
        print("\n📦 检查模型缓存...")
        cache_info = check_models_cache()
        
        if not cache_info['exists']:
            print("首次运行，设置模型缓存目录...")
            setup_models_cache()
        
        print_models_info()
    
    # 检查模块
    if not MODULES_OK:
        print("\n项目模块加载失败，请检查src目录")
        return
    
    # 查找输入文件
    print("\n查找输入视频...")
    input_file = find_input_video()
    if not input_file:
        return
    
    # 验证输入文件
    if not validate_input_file(input_file):
        return
    
    # 生成输出文件名
    output_file = generate_output_filename()
    
    print(f"\n配置信息:")
    print(f"  输入文件: {Path(input_file).name}")
    print(f"  输出文件: {output_file}")
    
    # 执行转录
    print(f"\n开始自动转录...")
    success = auto_transcript(input_file, output_file)
    
    if success:
        print(f"\n🎉 转录成功完成!")
    else:
        print(f"\n❌ 转录失败")


if __name__ == "__main__":
    main()