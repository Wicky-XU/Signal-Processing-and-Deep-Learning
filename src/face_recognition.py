#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸识别模块 - 真正的人脸特征对比识别
使用MTCNN和FaceNet进行人脸检测和特征提取，通过相似度计算确定说话者
支持本地模型缓存功能
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader


class FaceRecognizer:
    """真正的人脸识别器类 - 基于深度学习特征对比"""
    
    def __init__(self, database_path="data/input/test_images", 
                 models_cache_dir=r"C:\Users\29873\code\Summer-projects\auto-transcript\models"):
        """
        初始化人脸识别器
        
        Args:
            database_path: 人脸数据库路径
            models_cache_dir: 本地模型缓存目录
        """
        self.database_path = Path(database_path)
        self.models_cache_dir = Path(models_cache_dir)
        
        # 创建模型缓存目录
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"模型缓存目录: {self.models_cache_dir}")
        
        # 设置设备
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'人脸识别设备: {self.device}')
        
        # 设置模型缓存环境变量
        os.environ['TORCH_HOME'] = str(self.models_cache_dir)
        
        # 初始化MTCNN和FaceNet模型
        try:
            print("正在加载人脸识别模型...")
            
            # 检查本地是否已有模型
            mtcnn_cache = self.models_cache_dir / "mtcnn"
            facenet_cache = self.models_cache_dir / "facenet" 
            
            if mtcnn_cache.exists() and facenet_cache.exists():
                print("使用本地缓存的模型...")
            else:
                print("首次运行，正在下载模型到本地缓存...")
            
            self.mtcnn = MTCNN(
                image_size=160, 
                margin=0, 
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], 
                factor=0.709, 
                post_process=True,
                device=self.device
            )
            
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            # 保存模型到本地缓存
            self._save_models_to_cache()
            
            print("✅ MTCNN和FaceNet模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.mtcnn = None
            self.resnet = None
        
        # 人脸数据库
        self.face_database = {}
        self.load_face_database()
    
    def _save_models_to_cache(self):
        """保存模型到本地缓存"""
        try:
            # 保存FaceNet模型
            if self.resnet is not None:
                facenet_path = self.models_cache_dir / "inception_resnet_v1.pth"
                if not facenet_path.exists():
                    torch.save(self.resnet.state_dict(), facenet_path)
                    print(f"✅ FaceNet模型已保存到: {facenet_path}")
            
            # 创建模型信息文件
            info_file = self.models_cache_dir / "model_info.txt"
            with open(info_file, 'w', encoding='utf-8') as f:
                from datetime import datetime
                f.write("# Auto Transcript - 人脸识别模型缓存\n")
                f.write(f"# 创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# 模型列表:\n")
                f.write("# - inception_resnet_v1.pth: 人脸特征提取模型 (VGGFace2预训练)\n")
                f.write("# - MTCNN权重由facenet-pytorch自动管理\n")
                f.write(f"# 预计缓存大小: ~110MB\n")
            
        except Exception as e:
            print(f"模型缓存保存失败: {e}")
    
    def _load_models_from_cache(self):
        """从本地缓存加载模型"""
        try:
            facenet_path = self.models_cache_dir / "inception_resnet_v1.pth"
            
            if facenet_path.exists():
                print("从本地缓存加载FaceNet模型...")
                self.resnet.load_state_dict(torch.load(facenet_path, map_location=self.device))
                print("✅ 本地模型加载成功")
                return True
            else:
                print("本地缓存不存在，使用在线模型")
                return False
                
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            return False
    
    def get_cache_info(self):
        """获取模型缓存信息"""
        if not self.models_cache_dir.exists():
            return "模型缓存目录不存在"
        
        try:
            # 计算缓存大小
            total_size = 0
            file_count = 0
            
            for file_path in self.models_cache_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            size_mb = total_size / (1024 * 1024)
            
            info = f"""
模型缓存信息:
- 缓存目录: {self.models_cache_dir}
- 文件数量: {file_count}
- 总大小: {size_mb:.1f} MB
- 主要文件: inception_resnet_v1.pth, model_info.txt
            """
            
            return info.strip()
            
        except Exception as e:
            return f"获取缓存信息失败: {e}"
    
    def clear_cache(self):
        """清理模型缓存"""
        try:
            import shutil
            if self.models_cache_dir.exists():
                shutil.rmtree(self.models_cache_dir)
                print(f"✅ 模型缓存已清理: {self.models_cache_dir}")
            else:
                print("模型缓存目录不存在")
        except Exception as e:
            print(f"❌ 清理缓存失败: {e}")
    
    def load_face_database(self):
        """加载人脸数据库，提取每个人的人脸特征"""
        if not self.mtcnn or not self.resnet:
            print("模型未加载，跳过数据库加载")
            return
        
        if not self.database_path.exists():
            print(f"人脸数据库不存在: {self.database_path}")
            return
        
        print("加载人脸数据库...")
        
        try:
            def collate_fn(x):
                return x[0]
            
            # 加载图片数据集
            dataset = datasets.ImageFolder(str(self.database_path))
            dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
            loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)
            
            # 处理数据库中的每张人脸
            aligned_faces = []
            names = []
            
            for x, y in loader:
                person_name = dataset.idx_to_class[y]
                
                # 使用MTCNN检测和对齐人脸
                x_aligned, prob = self.mtcnn(x, return_prob=True)
                
                if x_aligned is not None and prob > 0.9:  # 置信度要求
                    aligned_faces.append(x_aligned)
                    names.append(person_name)
                    print(f"  处理 {person_name} 的照片，置信度: {prob:.3f}")
            
            if aligned_faces:
                # 计算人脸特征嵌入
                aligned_tensor = torch.stack(aligned_faces).to(self.device)
                with torch.no_grad():
                    embeddings = self.resnet(aligned_tensor).detach().cpu()
                
                # 按人员分组特征
                for name, embedding in zip(names, embeddings):
                    if name not in self.face_database:
                        self.face_database[name] = []
                    self.face_database[name].append(embedding)
                
                # 计算每个人的平均特征
                for name in self.face_database:
                    if len(self.face_database[name]) > 1:
                        # 多张照片取平均
                        avg_embedding = torch.stack(self.face_database[name]).mean(dim=0)
                        self.face_database[name] = avg_embedding
                    else:
                        # 单张照片直接使用
                        self.face_database[name] = self.face_database[name][0]
                    
                    print(f"  {name}: 特征向量准备完成")
                
                print(f"人脸数据库加载完成，共{len(self.face_database)}个人")
            else:
                print("未能从数据库中提取到有效人脸特征")
                
        except Exception as e:
            print(f"加载人脸数据库失败: {e}")
    
    def extract_face_from_video(self, video_path):
        """从视频中提取人脸特征"""
        if not self.mtcnn or not self.resnet:
            print("模型未加载，无法提取人脸特征")
            return None
        
        try:
            print("从视频中提取人脸特征...")
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            video_faces = []
            
            while cap.isOpened() and len(video_faces) < 5:  # 最多提取5张人脸
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:  # 每30帧处理一次
                    # 转换为RGB格式
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # 使用MTCNN检测和对齐人脸
                    face_aligned, prob = self.mtcnn(pil_image, return_prob=True)
                    
                    if face_aligned is not None and prob > 0.85:  # 较高的置信度要求
                        video_faces.append(face_aligned)
                        print(f"  提取人脸 {len(video_faces)}/5，置信度: {prob:.3f}")
            
            cap.release()
            
            if video_faces:
                # 计算视频人脸的特征嵌入
                video_tensor = torch.stack(video_faces).to(self.device)
                with torch.no_grad():
                    video_embeddings = self.resnet(video_tensor).detach().cpu()
                
                # 计算平均特征
                avg_video_embedding = video_embeddings.mean(dim=0)
                print(f"视频人脸特征提取完成，基于{len(video_faces)}张人脸")
                return avg_video_embedding
            else:
                print("未能从视频中提取到有效人脸")
                return None
                
        except Exception as e:
            print(f"视频人脸提取失败: {e}")
            return None
    
    def calculate_similarity(self, embedding1, embedding2):
        """计算两个人脸特征的相似度"""
        try:
            # 使用余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0), 
                embedding2.unsqueeze(0)
            ).item()
            return similarity
        except Exception as e:
            print(f"相似度计算失败: {e}")
            return 0.0
    
    def identify_speaker_by_similarity(self, video_embedding, similarity_threshold=0.6):
        """通过相似度对比识别说话者"""
        if not self.face_database:
            print("人脸数据库为空，无法进行对比")
            return "Unknown"
        
        if video_embedding is None:
            print("视频人脸特征为空，无法识别")
            return "Unknown"
        
        print("进行人脸相似度对比...")
        
        best_match = None
        best_similarity = 0.0
        
        # 与数据库中每个人进行对比
        for person_name, db_embedding in self.face_database.items():
            similarity = self.calculate_similarity(video_embedding, db_embedding)
            print(f"  与 {person_name} 的相似度: {similarity:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_name
        
        # 判断是否超过阈值
        if best_similarity > similarity_threshold:
            print(f"识别结果: {best_match} (相似度: {best_similarity:.4f})")
            return best_match
        else:
            print(f"相似度过低 (最高: {best_similarity:.4f})，无法确定身份")
            return "Unknown"
    
    def identify_speaker(self, video_path):
        """
        识别视频中的说话者 - 完整流程
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            str: 说话者姓名
        """
        try:
            # 检查模型是否可用
            if not self.mtcnn or not self.resnet:
                print("深度学习模型不可用，返回默认说话者")
                return "Speaker"
            
            # 从视频中提取人脸特征
            video_embedding = self.extract_face_from_video(video_path)
            
            # 通过相似度识别说话者
            speaker_name = self.identify_speaker_by_similarity(video_embedding)
            
            return speaker_name
            
        except Exception as e:
            print(f"说话者识别过程出错: {e}")
            return "Unknown"
        
        finally:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def test_face_recognition():
    """测试人脸识别模块"""
    print("测试真正的人脸识别功能...")
    
    recognizer = FaceRecognizer()
    
    # 显示缓存信息
    print(recognizer.get_cache_info())
    
    # 测试视频路径
    test_video = "data/input/input.mp4"
    
    if Path(test_video).exists():
        speaker = recognizer.identify_speaker(test_video)
        print(f"识别结果: {speaker}")
    else:
        print("测试视频不存在")


if __name__ == "__main__":
    test_face_recognition()
    
    def load_face_database(self):
        """加载人脸数据库，提取每个人的人脸特征"""
        if not self.mtcnn or not self.resnet:
            print("模型未加载，跳过数据库加载")
            return
        
        if not self.database_path.exists():
            print(f"人脸数据库不存在: {self.database_path}")
            return
        
        print("加载人脸数据库...")
        
        try:
            def collate_fn(x):
                return x[0]
            
            # 加载图片数据集
            dataset = datasets.ImageFolder(str(self.database_path))
            dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
            loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)
            
            # 处理数据库中的每张人脸
            aligned_faces = []
            names = []
            
            for x, y in loader:
                person_name = dataset.idx_to_class[y]
                
                # 使用MTCNN检测和对齐人脸
                x_aligned, prob = self.mtcnn(x, return_prob=True)
                
                if x_aligned is not None and prob > 0.9:  # 置信度要求
                    aligned_faces.append(x_aligned)
                    names.append(person_name)
                    print(f"  处理 {person_name} 的照片，置信度: {prob:.3f}")
            
            if aligned_faces:
                # 计算人脸特征嵌入
                aligned_tensor = torch.stack(aligned_faces).to(self.device)
                with torch.no_grad():
                    embeddings = self.resnet(aligned_tensor).detach().cpu()
                
                # 按人员分组特征
                for name, embedding in zip(names, embeddings):
                    if name not in self.face_database:
                        self.face_database[name] = []
                    self.face_database[name].append(embedding)
                
                # 计算每个人的平均特征
                for name in self.face_database:
                    if len(self.face_database[name]) > 1:
                        # 多张照片取平均
                        avg_embedding = torch.stack(self.face_database[name]).mean(dim=0)
                        self.face_database[name] = avg_embedding
                    else:
                        # 单张照片直接使用
                        self.face_database[name] = self.face_database[name][0]
                    
                    print(f"  {name}: 特征向量准备完成")
                
                print(f"人脸数据库加载完成，共{len(self.face_database)}个人")
            else:
                print("未能从数据库中提取到有效人脸特征")
                
        except Exception as e:
            print(f"加载人脸数据库失败: {e}")
    
    def extract_face_from_video(self, video_path):
        """从视频中提取人脸特征"""
        if not self.mtcnn or not self.resnet:
            print("模型未加载，无法提取人脸特征")
            return None
        
        try:
            print("从视频中提取人脸特征...")
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            video_faces = []
            
            while cap.isOpened() and len(video_faces) < 5:  # 最多提取5张人脸
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:  # 每30帧处理一次
                    # 转换为RGB格式
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # 使用MTCNN检测和对齐人脸
                    face_aligned, prob = self.mtcnn(pil_image, return_prob=True)
                    
                    if face_aligned is not None and prob > 0.85:  # 较高的置信度要求
                        video_faces.append(face_aligned)
                        print(f"  提取人脸 {len(video_faces)}/5，置信度: {prob:.3f}")
            
            cap.release()
            
            if video_faces:
                # 计算视频人脸的特征嵌入
                video_tensor = torch.stack(video_faces).to(self.device)
                with torch.no_grad():
                    video_embeddings = self.resnet(video_tensor).detach().cpu()
                
                # 计算平均特征
                avg_video_embedding = video_embeddings.mean(dim=0)
                print(f"视频人脸特征提取完成，基于{len(video_faces)}张人脸")
                return avg_video_embedding
            else:
                print("未能从视频中提取到有效人脸")
                return None
                
        except Exception as e:
            print(f"视频人脸提取失败: {e}")
            return None
    
    def calculate_similarity(self, embedding1, embedding2):
        """计算两个人脸特征的相似度"""
        try:
            # 使用余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0), 
                embedding2.unsqueeze(0)
            ).item()
            return similarity
        except Exception as e:
            print(f"相似度计算失败: {e}")
            return 0.0
    
    def identify_speaker_by_similarity(self, video_embedding, similarity_threshold=0.6):
        """通过相似度对比识别说话者"""
        if not self.face_database:
            print("人脸数据库为空，无法进行对比")
            return "Unknown"
        
        if video_embedding is None:
            print("视频人脸特征为空，无法识别")
            return "Unknown"
        
        print("进行人脸相似度对比...")
        
        best_match = None
        best_similarity = 0.0
        
        # 与数据库中每个人进行对比
        for person_name, db_embedding in self.face_database.items():
            similarity = self.calculate_similarity(video_embedding, db_embedding)
            print(f"  与 {person_name} 的相似度: {similarity:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_name
        
        # 判断是否超过阈值
        if best_similarity > similarity_threshold:
            print(f"识别结果: {best_match} (相似度: {best_similarity:.4f})")
            return best_match
        else:
            print(f"相似度过低 (最高: {best_similarity:.4f})，无法确定身份")
            return "Unknown"
    
    def identify_speaker(self, video_path):
        """
        识别视频中的说话者 - 完整流程
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            str: 说话者姓名
        """
        try:
            # 检查模型是否可用
            if not self.mtcnn or not self.resnet:
                print("深度学习模型不可用，返回默认说话者")
                return "Speaker"
            
            # 从视频中提取人脸特征
            video_embedding = self.extract_face_from_video(video_path)
            
            # 通过相似度识别说话者
            speaker_name = self.identify_speaker_by_similarity(video_embedding)
            
            return speaker_name
            
        except Exception as e:
            print(f"说话者识别过程出错: {e}")
            return "Unknown"
        
        finally:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def test_face_recognition():
    """测试人脸识别模块"""
    print("测试真正的人脸识别功能...")
    
    recognizer = FaceRecognizer()
    
    # 测试视频路径
    test_video = "data/input/input.mp4"
    
    if Path(test_video).exists():
        speaker = recognizer.identify_speaker(test_video)
        print(f"识别结果: {speaker}")
    else:
        print("测试视频不存在")


if __name__ == "__main__":
    test_face_recognition()