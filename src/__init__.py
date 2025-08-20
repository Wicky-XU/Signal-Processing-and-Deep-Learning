#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Transcript Project - 源代码模块包

这个包包含了自动转录项目的所有核心模块：
- face_recognition: 人脸识别模块
- speech_recognition: 语音识别模块
- translation: 翻译模块
- utils: 工具函数模块

使用示例:
    from src.face_recognition import FaceRecognizer
    from src.speech_recognition import SpeechRecognizer
    from src.translation import Translator
    from src.utils import check_dependencies
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 导入主要类，方便外部使用
try:
    from .face_recognition import FaceRecognizer
    from .speech_recognition import SpeechRecognizer
    from .translation import Translator
    from .utils import (
        setup_logging,
        validate_input_file,
        save_transcript_results,
        check_dependencies,
        print_project_banner,
        cleanup_temp_files
    )
    
    # 定义公开的API
    __all__ = [
        'FaceRecognizer',
        'SpeechRecognizer',
        'Translator',
        'setup_logging',
        'validate_input_file', 
        'save_transcript_results',
        'check_dependencies',
        'print_project_banner',
        'cleanup_temp_files'
    ]
    
    print("Auto Transcript 模块包加载成功")
    
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保所有依赖包已正确安装")
    __all__ = []

# 项目元信息
PROJECT_INFO = {
    'name': 'Auto Transcript Project',
    'version': __version__,
    'description': '自动视频转录系统 - 整合人脸识别、语音识别和翻译功能',
    'author': __author__,
    'email': __email__,
    'license': 'MIT',
    'keywords': ['speech recognition', 'face recognition', 'translation', 'video transcription'],
    'python_requires': '>=3.7',
    'dependencies': [
        'opencv-python>=4.5.0',
        'pillow>=8.0.0',
        'numpy>=1.20.0',
        'moviepy>=1.0.3',
        'pydub>=0.25.0',
        'SpeechRecognition>=3.8.0',
        'googletrans==4.0.0rc1'
    ]
}


def get_project_info():
    """获取项目信息"""
    return PROJECT_INFO.copy()


def print_module_info():
    """打印模块信息"""
    info = get_project_info()
    
    print("\n" + "="*60)
    print(f"{info['name']} v{info['version']}")
    print(info['description'])
    print("="*60)
    print("模块结构:")
    print("  ├── face_recognition.py    # 人脸识别模块")
    print("  ├── speech_recognition.py  # 语音识别模块") 
    print("  ├── translation.py         # 翻译模块")
    print("  ├── utils.py              # 工具函数模块")
    print("  └── __init__.py           # 模块初始化")
    print("="*60)
    print(f"作者: {info['author']}")
    print(f"许可证: {info['license']}")
    print("="*60)


if __name__ == "__main__":
    print_module_info()