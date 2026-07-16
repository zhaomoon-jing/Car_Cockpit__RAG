#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音识别模块 - 使用Whisper进行语音转文本
"""

import os
import json
import wave
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import numpy as np

# 导入项目配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WhisperASR:
    """Whisper语音识别器"""
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        """
        初始化语音识别器
        
        Args:
            model_name: Whisper模型名称
            device: 运行设备 ("cpu" 或 "cuda")
        """
        self.model_name = model_name or MODEL_CONFIG['asr_model']
        self.device = device
        self.model = None
        self.processor = None
        
        logger.info(f"初始化WhisperASR，模型: {self.model_name}，设备: {device}")
    
    def load_model(self):
        """加载Whisper模型"""
        try:
            import whisper
            
            logger.info(f"正在加载Whisper模型: {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Whisper模型加载成功")
            
        except ImportError:
            logger.error("请安装 openai-whisper: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"加载Whisper模型失败: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_path: Path, **kwargs) -> Dict[str, Any]:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            **kwargs: 转录参数
            
        Returns:
            Dict: 转录结果
        """
        if self.model is None:
            self.load_model()
        
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        logger.info(f"开始转录音频: {audio_path.name}")
        
        try:
            # 默认转录参数
            transcribe_kwargs = {
                "language": "zh",  # 中文
                "task": "transcribe",
                "fp16": False if self.device == "cpu" else True,
                "verbose": False
            }
            
            # 更新用户参数
            transcribe_kwargs.update(kwargs)
            
            # 执行转录
            result = self.model.transcribe(str(audio_path), **transcribe_kwargs)
            
            logger.info(f"音频转录完成: {audio_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"转录音频失败 {audio_path}: {str(e)}")
            raise
    
    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000, **kwargs) -> Dict[str, Any]:
        """
        转录音频字节数据
        
        Args:
            audio_bytes: 音频字节数据
            sample_rate: 采样率
            **kwargs: 转录参数
            
        Returns:
            Dict: 转录结果
        """
        if self.model is None:
            self.load_model()
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            # 将字节数据保存为WAV文件
            self._bytes_to_wav(audio_bytes, tmp_path, sample_rate)
            
            # 转录临时文件
            result = self.transcribe_audio(tmp_path, **kwargs)
            
            # 删除临时文件
            tmp_path.unlink()
            
            return result
    
    def _bytes_to_wav(self, audio_bytes: bytes, output_path: Path, sample_rate: int = 16000):
        """
        将字节数据保存为WAV文件
        
        Args:
            audio_bytes: 音频字节数据
            output_path: 输出文件路径
            sample_rate: 采样率
        """
        try:
            import soundfile as sf
            import io
            
            # 尝试使用soundfile读取
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            
            # 如果采样率不匹配，重采样
            if sr != sample_rate:
                from scipy import signal
                if len(audio_data.shape) > 1:  # 多声道
                    audio_data = audio_data.T
                    resampled = []
                    for channel in audio_data:
                        resampled_channel = signal.resample(
                            channel, 
                            int(len(channel) * sample_rate / sr)
                        )
                        resampled.append(resampled_channel)
                    audio_data = np.array(resampled).T
                else:  # 单声道
                    audio_data = signal.resample(
                        audio_data, 
                        int(len(audio_data) * sample_rate / sr)
                    )
            
            # 保存为WAV文件
            sf.write(str(output_path), audio_data, sample_rate)
            
        except Exception as e:
            logger.warning(f"使用soundfile处理音频失败，尝试备用方法: {str(e)}")
            self._bytes_to_wav_fallback(audio_bytes, output_path, sample_rate)
    
    def _bytes_to_wav_fallback(self, audio_bytes: bytes, output_path: Path, sample_rate: int = 16000):
        """
        备用方法：将字节数据保存为WAV文件
        
        Args:
            audio_bytes: 音频字节数据
            output_path: 输出文件路径
            sample_rate: 采样率
        """
        try:
            import wave
            import struct
            import numpy as np
            
            # 假设是16位PCM音频
            # 将字节转换为numpy数组
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # 归一化到[-1, 1]
            audio_normalized = audio_array.astype(np.float32) / 32768.0
            
            # 保存为WAV文件
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)   # 16位
                wav_file.setframerate(sample_rate)
                
                # 转换回16位整数
                audio_int16 = (audio_normalized * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
                
        except Exception as e:
            logger.error(f"保存WAV文件失败: {str(e)}")
            raise
    
    def realtime_transcribe(self, audio_stream, chunk_duration: float = 1.0):
        """
        实时转录音频流
        
        Args:
            audio_stream: 音频流生成器
            chunk_duration: 分块时长（秒）
            
        Yields:
            Dict: 实时转录结果
        """
        if self.model is None:
            self.load_model()
        
        logger.info("开始实时转录...")
        
        # 这里需要根据具体的音频流格式实现
        # 示例：处理PCM音频流
        for chunk_idx, audio_chunk in enumerate(audio_stream):
            try:
                # 转录当前分块
                result = self.transcribe_bytes(audio_chunk)
                
                yield {
                    "chunk_index": chunk_idx,
                    "text": result.get("text", ""),
                    "confidence": result.get("segments", [{}])[0].get("confidence", 0.0) if result.get("segments") else 0.0,
                    "timestamp": chunk_idx * chunk_duration
                }
                
            except Exception as e:
                logger.error(f"转录分块 {chunk_idx} 失败: {str(e)}")
                yield {
                    "chunk_index": chunk_idx,
                    "text": "",
                    "error": str(e),
                    "timestamp": chunk_idx * chunk_duration
                }
    
    def save_transcription(self, result: Dict[str, Any], output_path: Path):
        """
        保存转录结果
        
        Args:
            result: 转录结果
            output_path: 输出文件路径
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 提取关键信息
            transcription = {
                "text": result.get("text", ""),
                "language": result.get("language", ""),
                "segments": result.get("segments", []),
                "model": self.model_name,
                "device": self.device,
                "transcribed_at": str(Path(__file__).parent.parent / "speech_asr" / "transcriptions")
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False, indent=2)
            
            logger.info(f"转录结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存转录结果失败: {str(e)}")
            raise

class AudioPreprocessor:
    """音频预处理器"""
    
    @staticmethod
    def validate_audio_file(audio_path: Path) -> Tuple[bool, str]:
        """
        验证音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        if not audio_path.exists():
            return False, f"文件不存在: {audio_path}"
        
        # 检查文件格式
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        if audio_path.suffix.lower() not in supported_formats:
            return False, f"不支持的文件格式: {audio_path.suffix}，支持格式: {', '.join(supported_formats)}"
        
        # 检查文件大小（最大100MB）
        max_size = 100 * 1024 * 1024  # 100MB
        if audio_path.stat().st_size > max_size:
            return False, f"文件过大: {audio_path.stat().st_size / 1024 / 1024:.1f}MB > 100MB"
        
        return True, "文件有效"
    
    @staticmethod
    def convert_to_wav(input_path: Path, output_path: Path = None) -> Path:
        """
        转换音频文件为WAV格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径（可选）
            
        Returns:
            Path: 输出文件路径
        """
        try:
            import subprocess
            import tempfile
            
            if output_path is None:
                output_path = input_path.parent / f"{input_path.stem}_converted.wav"
            
            # 使用ffmpeg转换
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-y', str(output_path)
            ]
            
            logger.info(f"转换音频文件: {input_path.name} -> {output_path.name}")
            
            # 运行ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"音频转换成功: {output_path}")
                return output_path
            else:
                raise RuntimeError("转换失败: 输出文件为空或不存在")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg转换失败: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("请安装ffmpeg: sudo apt install ffmpeg 或从官网下载")
            raise
        except Exception as e:
            logger.error(f"音频转换失败: {str(e)}")
            raise

def main():
    """主函数"""
    print("=" * 50)
    print("Whisper语音识别工具 - 汽车座舱RAG系统")
    print("=" * 50)
    print(f"模型: {MODEL_CONFIG['asr_model']}")
    print("=" * 50)
    
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='Whisper语音识别')
    parser.add_argument('audio_file', type=str, help='音频文件路径')
    parser.add_argument('--output', type=str, default='transcription.json', help='输出文件路径')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='运行设备')
    parser.add_argument('--language', type=str, default='zh', help='语言代码')
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio_file)
    
    # 验证音频文件
    preprocessor = AudioPreprocessor()
    is_valid, message = preprocessor.validate_audio_file(audio_path)
    
    if not is_valid:
        print(f"❌ {message}")
        exit(1)
    
    try:
        # 初始化ASR
        asr = WhisperASR(device=args.device)
        
        # 转录音频
        print(f"正在转录: {audio_path.name}")
        result = asr.transcribe_audio(audio_path, language=args.language)
        
        # 保存结果
        output_path = Path(args.output)
        asr.save_transcription(result, output_path)
        
        # 显示结果
        print("\n📝 转录结果:")
        print("-" * 50)
        print(result.get("text", ""))
        print("-" * 50)
        print(f"\n✅ 转录完成! 结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"❌ 转录失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()