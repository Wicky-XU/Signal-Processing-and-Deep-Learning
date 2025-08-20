#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译模块
负责将识别出的文本翻译为英语
"""

from googletrans import Translator as GoogleTranslator


class Translator:
    """翻译器类"""
    
    def __init__(self):
        """初始化翻译器"""
        try:
            self.translator = GoogleTranslator()
            # 测试翻译功能
            test_result = self.translator.translate("Bonjour", src='fr', dest='en')
            print(f"Google翻译初始化成功，测试: 'Bonjour' -> '{test_result.text}'")
            self.available = True
        except Exception as e:
            print(f"翻译器初始化失败: {e}")
            self.translator = None
            self.available = False
    
    def translate_text(self, text, source_lang='fr', target_lang='en'):
        """
        翻译文本
        
        Args:
            text: 要翻译的文本
            source_lang: 源语言代码
            target_lang: 目标语言代码
            
        Returns:
            str: 翻译后的文本
        """
        if not self.available or not text:
            return text
        
        try:
            result = self.translator.translate(text, src=source_lang, dest=target_lang)
            return result.text
        except Exception as e:
            print(f"翻译失败: {e}")
            return text
    
    def process_transcript_item(self, item, speaker_name):
        """
        处理单个转录项
        
        Args:
            item: 转录项字典
            speaker_name: 说话者姓名
            
        Returns:
            str: 格式化的转录行
        """
        timestamp = item['timestamp']
        original_text = item['text']
        source_lang = item['language']
        
        # 翻译法语内容
        if source_lang == 'fr-FR':
            translated_text = self.translate_text(original_text, 'fr', 'en')
            print(f"翻译: {original_text} -> {translated_text}")
        else:
            translated_text = original_text
        
        return f"[{timestamp}] {speaker_name}: {translated_text}"
    
    def process_transcript(self, transcript_data, speaker_name):
        """
        处理完整的转录数据
        
        Args:
            transcript_data: 转录数据列表
            speaker_name: 说话者姓名
            
        Returns:
            list: 处理后的转录行列表
        """
        if not transcript_data:
            return []
        
        print("开始翻译处理...")
        results = []
        
        for item in transcript_data:
            line = self.process_transcript_item(item, speaker_name)
            results.append(line)
        
        print(f"翻译处理完成，共{len(results)}行")
        return results
    
    def get_language_name(self, language_code):
        """
        获取语言名称
        
        Args:
            language_code: 语言代码
            
        Returns:
            str: 语言名称
        """
        language_map = {
            'fr-FR': 'French',
            'en-US': 'English',
            'zh-CN': 'Chinese',
            'de-DE': 'German',
            'es-ES': 'Spanish'
        }
        return language_map.get(language_code, language_code)


def test_translation():
    """测试翻译模块"""
    translator = Translator()
    
    # 测试数据
    test_data = [
        {
            'timestamp': '00:00:00',
            'text': 'Bonjour le monde',
            'language': 'fr-FR'
        },
        {
            'timestamp': '00:00:05',
            'text': 'Comment allez-vous?',
            'language': 'fr-FR'
        }
    ]
    
    results = translator.process_transcript(test_data, "TestSpeaker")
    
    print("\n翻译测试结果:")
    for line in results:
        print(line)


if __name__ == "__main__":
    print("翻译模块测试")
    test_translation()