from collections import defaultdict
from transformers import AutoTokenizer
import time
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from functools import lru_cache
from typing import List
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


class TokenMatch:
    """Token matching class with rolling hash pool and multi-process support"""
    
    def __init__(self, window_size=5, chunk_size=256, max_workers=None, 
                 use_multi_process=True, length_threshold=256):
        """
        Initialize TokenMatch with a rolling hash pool
        
        Args:
            window_size: Size of the sliding window for matching
            chunk_size: Size of chunks for long sequence processing
            max_workers: Maximum number of worker processes
            use_multi_process: Whether to use multi-process matching
            length_threshold: Sequence length threshold for multi-process matching
        """
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.base = 256
        self.modulus = 1_000_000_007
        self.base_power = pow(self.base, window_size - 1, self.modulus)
        
        # Hash pool: {hash_value: [(token_sequence_id, start_position, data), ...]}
        self.hash_pool = {}
        # Token sequences pool: {sequence_id: (token_sequence, data)}
        self.token_pool = {}
        self.next_sequence_id = 0
        
        # 进程配置
        self.use_multi_process = use_multi_process
        self.length_threshold = length_threshold
        
        # 进程池
        if self.use_multi_process:
            self.max_workers = max_workers or mp.cpu_count()
            self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.process_pool = None
    
    def rolling_hash(self, tokens, start, prev_hash=None):
        """Calculate rolling hash for a window of tokens"""
        if prev_hash is None:
            hash_val = 0
            for i in range(self.window_size):
                hash_val = (hash_val * self.base + tokens[start + i]) % self.modulus
            return hash_val
        
        old_val = tokens[start - 1]
        new_val = tokens[start + self.window_size - 1]
        hash_val = ((prev_hash - old_val * self.base_power) * self.base + new_val) % self.modulus
        return hash_val
    
    def add_sequence(self, token_sequence, data=None):
        """
        Add a token sequence to the hash pool
        
        Args:
            token_sequence: List of token IDs to add
            data: Associated data for this token sequence
        """
        if len(token_sequence) < self.window_size:
            return
        
        sequence_id = self.next_sequence_id
        self.next_sequence_id += 1
        self.token_pool[sequence_id] = (token_sequence, data)
        
        # Calculate rolling hashes for all windows
        current_hash = None
        for i in range(len(token_sequence) - self.window_size + 1):
            current_hash = self.rolling_hash(token_sequence, i, current_hash)
            if current_hash in self.hash_pool:
                self.hash_pool[current_hash].append((sequence_id, i, data))
            else:
                self.hash_pool[current_hash] = [(sequence_id, i, data)]
    
    @staticmethod
    def _match_chunk_static(args):
        """Static method for processing a chunk of the sequence for matching"""
        target_chunk, candidate_chunk, candidate_id, chunk_start, window_size, base, modulus, base_power, hash_pool = args
        
        chunk_matches = []
        target_matched = set()
        candidate_matched = set()
        
        def rolling_hash(tokens, start, prev_hash=None):
            if prev_hash is None:
                hash_val = 0
                for i in range(window_size):
                    hash_val = (hash_val * base + tokens[start + i]) % modulus
                return hash_val
            
            old_val = tokens[start - 1]
            new_val = tokens[start + window_size - 1]
            hash_val = ((prev_hash - old_val * base_power) * base + new_val) % modulus
            return hash_val
        
        current_hash = None
        for i in range(len(target_chunk) - window_size + 1):
            current_hash = rolling_hash(target_chunk, i, current_hash)
            
            if current_hash in hash_pool:
                for seq_id, start_pos, data in hash_pool[current_hash]:
                    if seq_id != candidate_id:
                        continue
                    
                    # 确保候选序列中的位置在块范围内
                    if start_pos >= len(candidate_chunk) - window_size + 1:
                        continue
                    
                    if target_chunk[i:i+window_size] == candidate_chunk[start_pos:start_pos+window_size]:
                        match_length = window_size
                        while (i + match_length < len(target_chunk) and 
                               start_pos + match_length < len(candidate_chunk) and
                               target_chunk[i + match_length] == candidate_chunk[start_pos + match_length]):
                            match_length += 1
                        
                        for k in range(match_length):
                            t_idx = i + k
                            c_idx = start_pos + k
                            if (t_idx not in target_matched and 
                                c_idx not in candidate_matched and 
                                target_chunk[t_idx] == candidate_chunk[c_idx]):
                                # 添加全局位置和对应的data
                                chunk_matches.append((t_idx + chunk_start, c_idx + chunk_start, data))
                                target_matched.add(t_idx)
                                candidate_matched.add(c_idx)
        
        return chunk_matches
    
    def match(self, target_token_ids, candidate_token_ids):
        """
        Find matching tokens between target and candidate sequences
        
        Args:
            target_token_ids: Target token sequence
            candidate_token_ids: Candidate token sequence
        
        Returns:
            tuple: (target_matches, candidate_matches, match_data)
                - target_matches: List of matching positions in target sequence
                - candidate_matches: List of matching positions in candidate sequence
                - match_data: List of associated data for each match
        """
        if not target_token_ids or not candidate_token_ids:
            return [], [], []
        
        # Add candidate sequence to pool if not already present
        candidate_id = None
        for seq_id, (seq, _) in self.token_pool.items():
            if seq == candidate_token_ids:
                candidate_id = seq_id
                break
        
        if candidate_id is None:
            self.add_sequence(candidate_token_ids)
            candidate_id = self.next_sequence_id - 1
        
        # 根据配置决定是否使用多进程
        if not self.use_multi_process or len(target_token_ids) <= self.length_threshold:
            # 单进程匹配
            target_matches = []
            candidate_matches = []
            match_data = []
            target_matched = set()
            candidate_matched = set()
            
            current_hash = None
            for i in range(len(target_token_ids) - self.window_size + 1):
                current_hash = self.rolling_hash(target_token_ids, i, current_hash)
                
                if current_hash in self.hash_pool:
                    for seq_id, start_pos, data in self.hash_pool[current_hash]:
                        if seq_id != candidate_id:
                            continue
                        
                        if target_token_ids[i:i+self.window_size] == candidate_token_ids[start_pos:start_pos+self.window_size]:
                            match_length = self.window_size
                            while (i + match_length < len(target_token_ids) and 
                                   start_pos + match_length < len(candidate_token_ids) and
                                   target_token_ids[i + match_length] == candidate_token_ids[start_pos + match_length]):
                                match_length += 1
                            
                            for k in range(match_length):
                                t_idx = i + k
                                c_idx = start_pos + k
                                if (t_idx not in target_matched and 
                                    c_idx not in candidate_matched and 
                                    target_token_ids[t_idx] == candidate_token_ids[c_idx]):
                                    target_matches.append(t_idx)
                                    candidate_matches.append(c_idx)
                                    match_data.append(data)
                                    target_matched.add(t_idx)
                                    candidate_matched.add(c_idx)
        else:
            # 多进程匹配
            # 将序列分成多个块，确保块之间有重叠以避免边界问题
            overlap = self.window_size - 1
            target_chunks = []
            candidate_chunks = []
            chunk_starts = []
            
            for i in range(0, len(target_token_ids), self.chunk_size - overlap):
                chunk_end = min(i + self.chunk_size, len(target_token_ids))
                target_chunks.append(target_token_ids[i:chunk_end])
                candidate_chunks.append(candidate_token_ids[i:chunk_end])
                chunk_starts.append(i)
            
            # 准备任务参数
            tasks = []
            for i, (t_chunk, c_chunk) in enumerate(zip(target_chunks, candidate_chunks)):
                chunk_start = chunk_starts[i]
                tasks.append((t_chunk, c_chunk, candidate_id, chunk_start, 
                            self.window_size, self.base, self.modulus, 
                            self.base_power, self.hash_pool))
            
            # 使用进程池处理任务
            chunk_results = list(self.process_pool.map(self._match_chunk_static, tasks))
            
            # 合并结果，确保没有重复的匹配
            target_matches = []
            candidate_matches = []
            match_data = []
            target_matched = set()
            candidate_matched = set()
            
            for chunk_result in chunk_results:
                for t_idx, c_idx, data in chunk_result:
                    if t_idx not in target_matched and c_idx not in candidate_matched:
                        target_matches.append(t_idx)
                        candidate_matches.append(c_idx)
                        match_data.append(data)
                        target_matched.add(t_idx)
                        candidate_matched.add(c_idx)
            
            # 验证匹配的正确性
            for t_idx, c_idx in zip(target_matches, candidate_matches):
                if target_token_ids[t_idx] != candidate_token_ids[c_idx]:
                    print(f"Error: Mismatch at positions {t_idx} and {c_idx}")
                    print(f"Target token: {target_token_ids[t_idx]}")
                    print(f"Candidate token: {candidate_token_ids[c_idx]}")
                    raise AssertionError("Matching tokens are not identical")
        
        return target_matches, candidate_matches, match_data
    
    def clear_pool(self):
        """Clear the hash pool and token pool"""
        self.hash_pool.clear()
        self.token_pool.clear()
        self.next_sequence_id = 0
    
    def __del__(self):
        """Cleanup process pool"""
        if self.process_pool is not None:
            self.process_pool.shutdown()

def test_multi_process_performance():
    """测试多进程匹配的性能"""
    print("\n=== 测试多进程匹配性能 ===")
    
    def generate_test_case(length):
        """生成指定长度的测试序列"""
        return [random.randint(1, 100) for _ in range(length)]
    
    # 测试参数
    sequence_lengths = [256, 512, 1024, 2048, 4096]
    num_tests = 20
    chunk_sizes = [128, 256, 512]
    length_thresholds = [128, 256, 512]
    
    # 存储结果
    results = {cs: {'times': [], 'lengths': []} for cs in chunk_sizes}
    
    for chunk_size in chunk_sizes:
        print(f"\n测试块大小: {chunk_size}")
        print("序列长度\t平均时间(ms)\t标准差(ms)")
        print("-" * 50)
        
        # 测试不同长度阈值
        for threshold in length_thresholds:
            print(f"\n长度阈值: {threshold}")
            matcher = TokenMatch(window_size=3, 
                               chunk_size=chunk_size,
                               use_multi_process=True,
                               length_threshold=threshold)
            
            for seq_len in sequence_lengths:
                times = []
                
                for _ in range(num_tests):
                    # 生成测试序列
                    seq1 = generate_test_case(seq_len)
                    seq2 = generate_test_case(seq_len)
                    
                    # 插入一些相同的片段
                    for _ in range(3):
                        segment_len = random.randint(5, 10)
                        segment = generate_test_case(segment_len)
                        
                        pos1 = random.randint(0, seq_len - segment_len)
                        pos2 = random.randint(0, seq_len - segment_len)
                        
                        seq1[pos1:pos1 + segment_len] = segment
                        seq2[pos2:pos2 + segment_len] = segment
                    
                    # 测量时间
                    start_time = time.time()
                    matcher.add_sequence(seq1)
                    target_matches, candidate_matches, match_data = matcher.match(seq1, seq2)
                    end_time = time.time()
                    
                    times.append((end_time - start_time) * 1000)  # 转换为毫秒
                
                # 计算统计信息
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                results[chunk_size]['times'].append(avg_time)
                results[chunk_size]['lengths'].append(seq_len)
                
                print(f"{seq_len}\t\t{avg_time:.2f}\t\t{std_time:.2f}")
    
    # 绘制性能图表
    plt.figure(figsize=(12, 8))
    
    for chunk_size in chunk_sizes:
        plt.errorbar(results[chunk_size]['lengths'], 
                    results[chunk_size]['times'],
                    fmt='o-',
                    label=f'Chunk Size = {chunk_size}',
                    capsize=5)
    
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Multi-Process Matching Performance', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 设置x轴为对数刻度
    plt.xscale('log', base=2)
    
    # 添加次要网格线
    plt.grid(True, which="minor", linestyle=':', alpha=0.4)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('multi_process_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n多进程性能测试完成！图表已保存为 'multi_process_performance.png'")

def test_single_process_performance():
    """测试单进程匹配的性能"""
    print("\n=== 测试单进程匹配性能 ===")
    
    def generate_test_case(length):
        """生成指定长度的测试序列"""
        return [random.randint(1, 100) for _ in range(length)]
    
    # 测试参数
    sequence_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    num_tests = 50  # 每个长度测试50次
    window_sizes = [3, 5, 7]  # 测试不同的窗口大小
    
    # 存储结果
    results = {ws: {'times': [], 'lengths': []} for ws in window_sizes}
    
    for window_size in window_sizes:
        print(f"\n窗口大小: {window_size}")
        print("序列长度\t平均时间(ms)\t标准差(ms)")
        print("-" * 50)
        
        matcher = TokenMatch(window_size=window_size, 
                           use_multi_process=False)  # 使用单进程模式
        
        for seq_len in sequence_lengths:
            times = []
            
            for _ in range(num_tests):
                # 生成测试序列
                seq1 = generate_test_case(seq_len)
                seq2 = generate_test_case(seq_len)
                
                # 插入一些相同的片段
                for _ in range(3):
                    segment_len = random.randint(5, 10)
                    segment = generate_test_case(segment_len)
                    
                    pos1 = random.randint(0, seq_len - segment_len)
                    pos2 = random.randint(0, seq_len - segment_len)
                    
                    seq1[pos1:pos1 + segment_len] = segment
                    seq2[pos2:pos2 + segment_len] = segment
                
                # 测量时间
                start_time = time.time()
                matcher.add_sequence(seq1)
                target_matches, candidate_matches, match_data = matcher.match(seq1, seq2)
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            # 计算统计信息
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results[window_size]['times'].append(avg_time)
            results[window_size]['lengths'].append(seq_len)
            
            print(f"{seq_len}\t\t{avg_time:.2f}\t\t{std_time:.2f}")
    
    # 绘制性能图表
    plt.figure(figsize=(12, 8))
    
    for window_size in window_sizes:
        plt.errorbar(results[window_size]['lengths'], 
                    results[window_size]['times'],
                    fmt='o-',
                    label=f'Window Size = {window_size}',
                    capsize=5)
    
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Single-Process Matching Performance', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 设置x轴为对数刻度
    plt.xscale('log', base=2)
    
    # 添加次要网格线
    plt.grid(True, which="minor", linestyle=':', alpha=0.4)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('single_process_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n单进程性能测试完成！图表已保存为 'single_process_performance.png'")

def match_sequences(target_token_ids, candidate_token_ids, window_size=5):
    """
    计算两个token序列之间的匹配关系
    
    Args:
        target_token_ids: 目标token序列
        candidate_token_ids: 候选token序列
        window_size: 滑动窗口大小，用于控制匹配的最小长度
        
    Returns:
        tuple: (target_matches, candidate_matches)
            - target_matches: 目标序列中匹配位置的列表
            - candidate_matches: 候选序列中对应匹配位置的列表
    """
    if not target_token_ids or not candidate_token_ids:
        return [], []
    
    # 哈希算法参数
    base = 256
    modulus = 1_000_000_007
    base_power = pow(base, window_size - 1, modulus)
    
    def rolling_hash(tokens, start, prev_hash=None):
        """计算滑动窗口的哈希值"""
        if prev_hash is None:
            hash_val = 0
            for i in range(window_size):
                hash_val = (hash_val * base + tokens[start + i]) % modulus
            return hash_val
        
        old_val = tokens[start - 1]
        new_val = tokens[start + window_size - 1]
        hash_val = ((prev_hash - old_val * base_power) * base + new_val) % modulus
        return hash_val
    
    # 初始化结果
    target_matches = []
    candidate_matches = []
    target_matched = set()
    candidate_matched = set()
    
    # 构建目标序列的哈希索引
    target_hash_index = {}
    current_hash = None
    for i in range(len(target_token_ids) - window_size + 1):
        current_hash = rolling_hash(target_token_ids, i, current_hash)
        if current_hash in target_hash_index:
            target_hash_index[current_hash].append(i)
        else:
            target_hash_index[current_hash] = [i]
    
    # 在候选序列中查找匹配
    current_hash = None
    for j in range(len(candidate_token_ids) - window_size + 1):
        current_hash = rolling_hash(candidate_token_ids, j, current_hash)
        if current_hash in target_hash_index:
            for i in target_hash_index[current_hash]:
                if target_token_ids[i:i+window_size] == candidate_token_ids[j:j+window_size]:
                    # 找到匹配后，尝试扩展匹配长度
                    match_length = window_size
                    while (i + match_length < len(target_token_ids) and 
                           j + match_length < len(candidate_token_ids) and
                           target_token_ids[i + match_length] == candidate_token_ids[j + match_length]):
                        match_length += 1
                    
                    # 记录匹配位置，确保对应位置的token相同且未被匹配过
                    for k in range(match_length):
                        t_idx = i + k
                        c_idx = j + k
                        if (t_idx not in target_matched and 
                            c_idx not in candidate_matched and 
                            target_token_ids[t_idx] == candidate_token_ids[c_idx]):
                            target_matches.append(t_idx)
                            candidate_matches.append(c_idx)
                            target_matched.add(t_idx)
                            candidate_matched.add(c_idx)
    
    # 验证结果
    assert len(target_matches) == len(candidate_matches), "匹配列表长度不一致"
    for t_idx, c_idx in zip(target_matches, candidate_matches):
        assert target_token_ids[t_idx] == candidate_token_ids[c_idx], "对应位置的token不匹配"
    
    return target_matches, candidate_matches

def test_match_sequences():
    """测试match_sequences函数的正确性"""
    print("\n=== 测试序列匹配函数 ===")
    
    def generate_test_case(length):
        """生成指定长度的测试序列"""
        return [random.randint(1, 100) for _ in range(length)]
    
    # 测试参数
    sequence_lengths = [32, 64, 128, 256, 512]
    num_tests = 50
    window_sizes = [3, 5, 7]
    
    for window_size in window_sizes:
        print(f"\n窗口大小: {window_size}")
        print("序列长度\t平均时间(ms)\t标准差(ms)")
        print("-" * 50)
        
        for seq_len in sequence_lengths:
            times = []
            
            for _ in range(num_tests):
                # 生成测试序列
                seq1 = generate_test_case(seq_len)
                seq2 = generate_test_case(seq_len)
                
                # 插入一些相同的片段
                for _ in range(3):
                    segment_len = random.randint(5, 10)
                    segment = generate_test_case(segment_len)
                    
                    pos1 = random.randint(0, seq_len - segment_len)
                    pos2 = random.randint(0, seq_len - segment_len)
                    
                    seq1[pos1:pos1 + segment_len] = segment
                    seq2[pos2:pos2 + segment_len] = segment
                
                # 测量时间
                start_time = time.time()
                target_matches, candidate_matches = match_sequences(seq1, seq2, window_size)
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            # 计算统计信息
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"{seq_len}\t\t{avg_time:.2f}\t\t{std_time:.2f}")

if __name__=="__main__":
    # 运行序列匹配测试
    test_match_sequences()
    
    # 运行单进程性能测试
    # test_single_process_performance()
    
    # # 运行多进程性能测试
    # test_multi_process_performance()
    

