#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音识别模块
负责从视频中提取和识别语音内容
"""

import os
from pathlib import Path
from datetime import timedelta
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
import speech_recognition as sr


class SpeechRecognizer:
    """语音识别器类"""
    
    def __init__(self, temp_dir="data/temp"):
        """
        初始化语音识别器
        
        Args:
            temp_dir: 临时文件目录
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.recognizer = sr.Recognizer()
        print("语音识别器初始化完成")
    
    def extract_audio_from_video(self, video_path):
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            str: 音频文件路径
        """
        try:
            print("从视频中提取音频...")
            clip = VideoFileClip(video_path)
            audio_path = self.temp_dir / "extracted_audio.wav"
            clip.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            clip.close()
            
            print(f"音频提取完成: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            print(f"音频提取失败: {e}")
            return None
    
    def split_audio_by_silence(self, audio_path):
        """
        根据静音分割音频
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            tuple: (音频片段列表, 时间戳列表)
        """
        try:
            print("分割音频...")
            audio = AudioSegment.from_file(audio_path)
            print(f"音频总长度: {len(audio)/1000:.1f}秒")
            
            # 静音分割
            chunks = silence.split_on_silence(
                audio,
                min_silence_len=800,    # 最小静音长度800ms
                silence_thresh=-38,     # 静音阈值-38dB
                keep_silence=300        # 保留300ms静音
            )
            
            # 如果分割失败，使用时间分割
            if not chunks or len(chunks) < 2:
                chunk_length = 8000  # 8秒
                chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
                print(f"使用时间分割，分割出{len(chunks)}个片段")
            else:
                print(f"使用静音分割，分割出{len(chunks)}个片段")
            
            # 计算时间戳
            timestamps = []
            current_time = 0
            for chunk in chunks:
                timestamps.append(current_time)
                current_time += len(chunk)
            
            return chunks, timestamps
            
        except Exception as e:
            print(f"音频分割失败: {e}")
            return [], []
    
    def recognize_audio_chunk(self, chunk, chunk_index, timestamp):
        """
        识别单个音频片段
        
        Args:
            chunk: 音频片段
            chunk_index: 片段索引
            timestamp: 时间戳
            
        Returns:
            dict: 识别结果
        """
        # 过滤过短的片段
        if len(chunk) < 1000:
            print(f"  片段{chunk_index+1}太短({len(chunk)}ms)，跳过")
            return None
        
        chunk_file = self.temp_dir / f"chunk_{chunk_index}.wav"
        
        try:
            print(f"处理片段 {chunk_index+1}")
            
            # 音频预处理
            processed_chunk = chunk
            if chunk.dBFS < -25:
                gain = min(15, abs(chunk.dBFS + 20))
                processed_chunk = chunk + gain
            
            # 导出临时文件
            processed_chunk.export(str(chunk_file), format="wav")
            
            # 语音识别
            with sr.AudioFile(str(chunk_file)) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                audio_data = self.recognizer.record(source)
                
                # 尝试识别法语
                text = None
                for attempt in range(2):
                    try:
                        text = self.recognizer.recognize_google(audio_data, language="fr-FR")
                        if text and len(text.strip()) > 2:
                            print(f"  识别成功: {text}")
                            break
                    except sr.UnknownValueError:
                        if attempt == 0:
                            continue
                        else:
                            print(f"  片段{chunk_index+1}无法识别")
                    except sr.RequestError as e:
                        print(f"  API错误: {e}")
                        break
                
                if text and len(text.strip()) > 2:
                    # 格式化时间戳
                    start_time = str(timedelta(seconds=timestamp//1000))
                    if len(start_time.split(':')) == 2:
                        start_time = "0:" + start_time
                    
                    return {
                        'timestamp': start_time,
                        'text': text.strip(),
                        'language': 'fr-FR'
                    }
                
                return None
                
        except Exception as e:
            print(f"  片段{chunk_index+1}处理失败: {e}")
            return None
        
        finally:
            # 清理临时文件
            if chunk_file.exists():
                chunk_file.unlink()
    
    def transcribe_video(self, video_path):
        """
        转录视频中的语音
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            list: 转录结果列表
        """
        try:
            # 提取音频
            audio_path = self.extract_audio_from_video(video_path)
            if not audio_path:
                return []
            
            # 分割音频
            chunks, timestamps = self.split_audio_by_silence(audio_path)
            if not chunks:
                return []
            
            # 识别每个片段
            results = []
            for i, (chunk, timestamp) in enumerate(zip(chunks, timestamps)):
                result = self.recognize_audio_chunk(chunk, i, timestamp)
                if result:
                    results.append(result)
            
            # 清理音频文件
            if Path(audio_path).exists():
                os.remove(audio_path)
            
            print(f"语音识别完成，共{len(results)}个有效片段")
            return results
            
        except Exception as e:
            print(f"语音转录失败: {e}")
            return []


def test_speech_recognition():
    """测试语音识别模块"""
    recognizer = SpeechRecognizer()
    
    # 测试视频路径
    test_video = "data/input/input.mp4"
    
    if Path(test_video).exists():
        results = recognizer.transcribe_video(test_video)
        
        print("\n语音识别测试结果:")
        for result in results:
            print(f"[{result['timestamp']}] {result['text']} ({result['language']})")
    else:
        print("测试视频不存在")


if __name__ == "__main__":
    print("语音识别模块测试")
    test_speech_recognition()