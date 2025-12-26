#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU检测脚本 - 验证HuggingFace模型是否在GPU上运行
"""

import torch
import psutil
import time
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

def check_gpu_availability():
    """检查GPU可用性"""
    print("=== GPU可用性检查 ===")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        
        # 显示每个GPU的信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            print(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f} GB")
            
            # 检查当前GPU内存使用情况
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  - 已分配: {allocated:.2f} GB, 缓存: {cached:.2f} GB")
    else:
        print("警告: 没有检测到可用的GPU，模型将在CPU上运行")
    
    print()
    return cuda_available

def test_huggingface_embeddings():
    """测试HuggingFace嵌入模型运行设备"""
    print("=== HuggingFace嵌入模型测试 ===")
    
    # 测试不同的设备配置
    device_configs = [
        ('auto', '自动选择'),
        ('cpu', '强制CPU'),
        ('cuda', '强制GPU'),
        ('cuda:0', 'GPU设备0')
    ]
    
    for device, description in device_configs:
        print(f"\n测试配置: {description} (device='{device}')")
        
        try:
            # 创建嵌入模型
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={'device': device}
            )
            
            # 测试文本
            test_text = "这是一个测试文本，用于验证模型运行设备"
            
            # 测量推理时间
            start_time = time.time()
            
            # 生成嵌入
            vector = embeddings.embed_query(test_text)
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # 毫秒
            
            print(f"  - 推理时间: {inference_time:.2f} ms")
            print(f"  - 向量维度: {len(vector)}")
            print(f"  - 向量示例: {vector[:5]}...")  # 显示前5个维度
            
            # 检查模型实际运行的设备
            if hasattr(embeddings.client, 'device'):
                print(f"  - 模型实际设备: {embeddings.client.device}")
            
        except Exception as e:
            print(f"  - 错误: {e}")

def monitor_gpu_usage():
    """监控GPU使用情况"""
    print("\n=== GPU使用情况监控 ===")
    
    if torch.cuda.is_available():
        print("开始监控GPU使用情况（5秒）...")
        
        for i in range(5):
            # 获取GPU内存使用情况
            for gpu_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                cached = torch.cuda.memory_reserved(gpu_id) / 1024**3
                
                print(f"GPU {gpu_id}: 已分配 {allocated:.2f} GB, 缓存 {cached:.2f} GB")
            
            # 获取CPU和内存使用情况
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            print(f"CPU使用率: {cpu_percent}%, 内存使用: {memory.percent}%")
            print("-" * 50)
            
            time.sleep(1)
    else:
        print("没有GPU可用，跳过GPU监控")

def benchmark_performance():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    # 测试文本列表
    test_texts = [
        "这是一个短文本",
        "这是一个中等长度的测试文本，用于验证模型性能",
        "这是一个较长的测试文本，包含更多的内容，用于全面测试模型的嵌入生成能力和性能表现"
    ]
    
    # 测试不同设备配置
    devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
    
    for device in devices:
        print(f"\n设备: {device}")
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={'device': device}
            )
            
            for i, text in enumerate(test_texts):
                start_time = time.time()
                
                # 多次测试取平均值
                times = []
                for _ in range(3):
                    single_start = time.time()
                    vector = embeddings.embed_query(text)
                    single_end = time.time()
                    times.append((single_end - single_start) * 1000)
                
                avg_time = sum(times) / len(times)
                
                print(f"  文本{i+1} ({len(text)}字符): {avg_time:.2f} ms")
                
        except Exception as e:
            print(f"  错误: {e}")

if __name__ == "__main__":
    print("HuggingFace GPU检测脚本")
    print("=" * 60)
    
    # 检查GPU可用性
    gpu_available = check_gpu_availability()
    
    # 测试HuggingFace嵌入模型
    test_huggingface_embeddings()
    
    # 监控GPU使用情况
    monitor_gpu_usage()
    
    # 性能基准测试
    benchmark_performance()
    
    print("\n=== 检测完成 ===")
    
    # 给出建议
    if gpu_available:
        print("✅ 检测到GPU，建议使用 device='cuda' 以获得最佳性能")
    else:
        print("⚠️  未检测到GPU，模型将在CPU上运行，建议使用 device='cpu'")
    
    # 检查当前main.py中的配置
    print("\n当前main.py配置检查:")
    print("  - 配置: model_kwargs={'device': 'gpu'}")
    print("  - 建议: 使用 'cuda' 而不是 'gpu' (HuggingFace标准命名)")
    print("  - 或者: 使用 'auto' 让库自动选择最佳设备")