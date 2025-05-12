import torch
from collections import deque
from typing import List, Optional, Union
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import MultiModalData
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter
from vllm import LLM
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import MilvusClient
import time
import os
from edit2 import TokenMatch, match_sequences
import uuid
os.environ["VLLM_USE_MODELSCOPE"] = "False"
import json
import threading
import queue
import multiprocessing as mp
import numpy as np

class Request:
    
    """
    请求类，包含请求的prompt和response
    """
    first_schedule_time: float # 提交请求的时间
    first_token_output_time: float # 第一个token输出时间 计算TTFT
    last_token_output_time: float # 最后一个token输出时间 计算吞吐量
    
    prompt_tokens: List[str] = []
    output_tokens: List[str] = []
    
    def __init__(self, prompt: str):
        self.prompt = prompt
        
        

    
class ShareEngine(LLMEngine):
    """
    KVShareCachePool is a class that shares the cache between the model and the inference server.
    1. 使用Text Embeeding模型，将输入的文本转换为向量，并使用Faiss存储在内存中。
    2. 使用多进程，查询文本token可复用片段 时间复杂度: O(n)/num_process
    """
    
class ShareLLM(LLM):
    """
    可以共享KVcache的LLM
    """
    def __init__(self, 
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,):
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = ShareEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()

        # 两个GPU 一个做LLM 一个做Sentence Embedding
        self.sentence_llm = SentenceTransformer("all-MiniLM-L6-v2",device="cuda:1")
        # 初始化Milvus
        self.kvcache_save_path = "kvcache_disk_pool/"
        os.makedirs(self.kvcache_save_path, exist_ok=True)
        self.client = MilvusClient(os.path.join(self.kvcache_save_path,"kvcache.db"))
        self.collection_name = "kvcache"
        
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.sentence_llm.get_sentence_embedding_dimension(),  # The vectors we will use in this demo has 768 dimensions
        )
        self.ttft_recoders = {}
        self.tpot_recoders = {}
        self.input_token_recorders = {}
        self.output_token_recorders = {}
        self.tokenizer = self.get_tokenizer()
        self.engine_running = True
    
    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> List[RequestOutput]:
        """
        这一个批次只能做KV Cache预计算生成
        """
        # cache_fuse_metadata['collect'] = False
        # cache_fuse_metadata['check'] = False  
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if self.llm_engine.model_config.skip_tokenizer_init \
            and prompts is not None:
            raise ValueError("prompts must be None if skip_tokenizer_init "
                             "is True")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")

        if prompts is not None:
            num_requests = len(prompts)
        else:
            assert prompt_token_ids is not None
            num_requests = len(prompt_token_ids)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        elif isinstance(sampling_params,
                        list) and len(sampling_params) != num_requests:
            raise ValueError("The lengths of prompts and sampling_params "
                             "must be the same.")
        if multi_modal_data:
            multi_modal_data.data = multi_modal_data.data.to(torch.float16)

        # Add requests to the engine.
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[
                i]
            self._add_request(
                prompt,
                sampling_params[i]
                if isinstance(sampling_params, list) else sampling_params,
                token_ids,
                lora_request=lora_request,
                # Get ith image while maintaining the batch dim.
                multi_modal_data=MultiModalData(
                    type=multi_modal_data.type,
                    data=multi_modal_data.data[i].unsqueeze(0))
                if multi_modal_data else None,
            )
        # return self._run_engine(use_tqdm)
    
    def precompute_generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> List[RequestOutput]:
        """
        这一个批次只能做KV Cache预计算生成
        """
        cache_fuse_metadata['collect'] = True
        cache_fuse_metadata['check'] = False
        self.generate(prompts, sampling_params, prompt_token_ids, use_tqdm, lora_request, multi_modal_data)
        outputs = self.sync_run_engine(use_tqdm=False)
        batch_text_embedding = self.sentence_llm.encode(prompts)
        for idx, output in enumerate(outputs):
            text_embedding = batch_text_embedding[idx]
            # ID 是当前时间戳x1000000+idx
            id = int(time.time()*1000000) + idx
            kvcache_disk_path = os.path.join(self.kvcache_save_path,f"{str(uuid.uuid4())}.pt")
            torch.save(output.hack_kvs, kvcache_disk_path)
            self.client.upsert(
                collection_name=self.collection_name,
                 data = {
                    "id": id,
                    "vector":text_embedding,
                    "prompt": prompts[idx],
                    "kvcache_disk_path": kvcache_disk_path
                }
            )

    def fr_generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
    ) -> List[RequestOutput]:
        """
        这个函数是用来生成请求的
        """
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['check'] = False
        self.generate(prompts, sampling_params, prompt_token_ids, use_tqdm=False)
    
    def share_generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> List[RequestOutput]:
        """
        这一个批次全部都需要做部分token的计算
        """
        # =================== HACK 这里预计算好所有信息,在ADD_REQUEST的时候,直接使用
        # cache_fuse_metadata['collect'] = False
        # cache_fuse_metadata['check'] = True
        
        # 在这里计算出全部的可以复用KV的位置
        batch_text_embedding = self.sentence_llm.encode(prompts)
        # 强制reshape
        batch_text_embedding = batch_text_embedding.reshape(len(prompts),-1)
        # 查询相似请求的KVCache
        results = self.client.search(
            collection_name=self.collection_name,
            data=batch_text_embedding,
            limit=1,
            output_fields=["prompt","kvcache_disk_path"]
        )
        # dtype = torch.bfloat16 if self.llm_engine.model_config.dtype == "bfloat16" else torch.float16
        # recompute: 0 ,
        current_request_id_start = self.request_counter.counter
        
        # 添加进度条
        for idx, result in tqdm(enumerate(results), total=len(results), desc="处理KV Cache匹配"):
            if (len(result) > 0 and result[0]["distance"] >= 0.5 and cache_fuse_metadata['recompute_mode'] == 1) or \
                (len(result) > 0 and result[0]["distance"] >= 0.5 and cache_fuse_metadata['recompute_mode'] in [0,2]):
                kvcache_disk_path = result[0]["kvcache_disk_path"]
                candidate_prompt = result[0]["prompt"]
                target_prompt = prompts[idx]
                
                # 将prompt转换为token_ids
                target_token_ids = self.tokenizer.encode(target_prompt)
                candidate_token_ids = self.tokenizer.encode(candidate_prompt)
                candidate_kvcache = torch.load(kvcache_disk_path)
                target_kvcache = torch.zeros([candidate_kvcache.shape[0],candidate_kvcache.shape[1],len(target_token_ids),candidate_kvcache.shape[-1]],device="cpu",dtype=candidate_kvcache.dtype)
                target_matched_idx, candidate_matched_idx = match_sequences(target_token_ids, candidate_token_ids)
                
                
                
                target_kvcache[:,:,target_matched_idx,:] = candidate_kvcache[:,:,candidate_matched_idx,:]
                
                if len(target_token_ids) -1 in target_matched_idx:
                    target_matched_idx.remove(len(target_token_ids) -1)
                
                self.llm_engine.model_executor.driver_worker.model_runner.cpu_prefetch_kvcache_pool[current_request_id_start+idx] = {
                    "kvcache": target_kvcache,
                    "reused_positions": [i for i in range(len(target_token_ids)) if i in target_matched_idx],
                    "unreused_positions": [i for i in range(len(target_token_ids)) if i not in target_matched_idx]
                }
            else:
                target_prompt = prompts[idx]
                target_token_ids = self.tokenizer.encode(target_prompt)
                # NOTE 注意修改变量
                target_kvcache = torch.zeros([32,2,len(target_token_ids),1024],device="cpu",dtype=torch.bfloat16)
                self.llm_engine.model_executor.driver_worker.model_runner.cpu_prefetch_kvcache_pool[current_request_id_start+idx] = {
                    "kvcache": target_kvcache,
                    "reused_positions": [],
                    "unreused_positions": [i for i in range(len(target_token_ids))]
                }
                

        self.generate(prompts, sampling_params, prompt_token_ids, use_tqdm, lora_request, multi_modal_data)

    
    def calculate_metrics(self, outputs: List[RequestOutput], dur_s: float) -> dict:
        """计算性能指标"""
        completed = 0
        total_input = 0
        total_output = 0
        ttfts = []
        tpots = []
        
        for output in outputs:
            if output.finished:
                completed += 1
                # 计算输入token数
                input_tokens = len(self.tokenizer.encode(output.prompt))
                total_input += input_tokens
                
                # 计算输出token数
                output_tokens = len(self.tokenizer.encode(output.outputs[0].text))
                total_output += output_tokens
                
                # 计算TTFT
                ttft = (output.metrics.first_token_time - output.metrics.first_scheduled_time) * 1000  # 转换为毫秒
                ttfts.append(ttft)
                
                # 计算TPOT (Time per Output Token)
                if output_tokens > 1:
                    # 确保时间差为正数
                    time_diff = max(0, output.metrics.finished_time - output.metrics.first_token_time)
                    tpot = time_diff / (output_tokens - 1) * 1000  # 转换为毫秒
                    if tpot > 0:  # 只记录有效的TPOT值
                        tpots.append(tpot)
        
        metrics = {
            "completed": completed,
            "total_input": total_input,
            "total_output": total_output,
            "request_throughput": completed / dur_s,
            "input_throughput": total_input / dur_s,
            "output_throughput": total_output / dur_s,
            "mean_ttft_ms": np.mean(ttfts) if ttfts else 0,
            "median_ttft_ms": np.median(ttfts) if ttfts else 0,
            "p99_ttft_ms": np.percentile(ttfts, 99) if ttfts else 0,
            "mean_tpot_ms": np.mean(tpots) if tpots else 0,
            "median_tpot_ms": np.median(tpots) if tpots else 0,
            "p99_tpot_ms": np.percentile(tpots, 99) if tpots else 0,
        }
        
        return metrics

    def _run_engine(self, use_tqdm: bool, rate: float = 10.0, requests: List[str] = None) -> List[RequestOutput]:
        """
        运行引擎处理请求，根据设定的QPS控制请求发送速率
        
        Args:
            use_tqdm: 是否显示进度条
            rate: 目标QPS (每秒请求数)
            requests: 待处理的请求列表
        """
        # 初始化进度条
        if use_tqdm and requests:
            pbar = tqdm(total=len(requests), desc="处理请求进度")
        
        # 存储输出结果
        outputs: List[RequestOutput] = []
        total_requests = len(requests) if requests else 0
        current_request_idx = 0  # 当前处理到的请求索引
        
        # 记录开始时间
        start_time = time.time()
        last_step_duration = 0.0  # 上一次step的运行时间
        time_credit = 0.0  # 累积的时间额度，用于控制请求发送
        
        while True:
            # 检查是否所有请求都已处理完毕
            if requests and current_request_idx >= total_requests and self.llm_engine.get_num_unfinished_requests() == 0:
                break
                
            prompts = []
            
            # 决定获取多少请求
            if requests and current_request_idx < total_requests:
                # 累积时间额度
                time_credit += last_step_duration
                
                # 计算累积时间内应该发送的请求数
                # 例如：如果rate=10，累积了0.15秒，则应该发送1个请求
                expected_requests = int(time_credit * rate)
                
                if expected_requests >= 1:
                    # 如果累积时间足够发送至少一个请求
                    batch_size = min(expected_requests, total_requests - current_request_idx)
                    if batch_size > 0:
                        prompts = requests[current_request_idx:current_request_idx + batch_size]
                        current_request_idx += batch_size
                        # 重置时间额度，但保留余数
                        # 例如：如果rate=10，发送了1个请求，剩余0.05秒
                        time_credit = (time_credit * rate - batch_size) / rate
                else:
                    # 如果累积时间不足一个请求，继续等待
                    prompts = []
            
            # 处理请求
            if prompts:
                # self.generate(
                #     prompts=prompts,
                #     sampling_params=SamplingParams(max_tokens=512, temperature=0.0)
                # )
                self.cacheblend_generate(
                    prompts=prompts,
                    sampling_params=SamplingParams(max_tokens=512, temperature=0.0)
                )
            cache_fuse_metadata['collect'] = False
            cache_fuse_metadata['check'] = True
            
            # 记录step开始时间
            step_start_time = time.time()
            
            # 处理引擎中的请求
            step_outputs = self.llm_engine.step()
            
            # 记录step结束时间并计算处理时间
            step_end_time = time.time()
            last_step_duration = step_end_time - step_start_time
            
            # 处理完成的请求
            for output in step_outputs:
                if output.finished:
                    # 清理KV cache
                    if output.request_id in self.llm_engine.model_executor.driver_worker.model_runner.cpu_hack_kvcache_pool:
                        output.hack_kvs = self.llm_engine.model_executor.driver_worker.model_runner.cpu_hack_kvcache_pool[output.request_id]
                        self.llm_engine.model_executor.driver_worker.model_runner.cpu_hack_kvcache_pool.pop(output.request_id)
                    # output.metrics.first_scheduled_time = self.llm_engine.model_executor.driver_worker.model_runner.real_schedule_time[output.request_id]
                    # output.metrics.first_token_time = self.llm_engine.model_executor.driver_worker.model_runner.real_first_token_time[output.request_id]
                    outputs.append(output)
                    if use_tqdm and requests:
                        pbar.update(1)
        
        if use_tqdm and requests:
            pbar.close()
            
        # 计算总运行时间（只包含step的时间）
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 计算性能指标
        metrics = self.calculate_metrics(outputs, total_duration)
        
        # 打印性能指标
        print("\n性能指标统计:")
        print(f"总运行时间: {total_duration:.2f}秒")
        print(f"完成请求数: {metrics['completed']}")
        print(f"总输入token数: {metrics['total_input']}")
        print(f"总输出token数: {metrics['total_output']}")
        print(f"请求吞吐量: {metrics['request_throughput']:.2f} req/s")
        print(f"输入token吞吐量: {metrics['input_throughput']:.2f} tokens/s")
        print(f"输出token吞吐量: {metrics['output_throughput']:.2f} tokens/s")
        print("\nTTFT统计:")
        print(f"平均TTFT: {metrics['mean_ttft_ms']:.2f}ms")
        print(f"中位数TTFT: {metrics['median_ttft_ms']:.2f}ms")
        print(f"P99 TTFT: {metrics['p99_ttft_ms']:.2f}ms")
        print("\nTPOT统计:")
        print(f"平均TPOT: {metrics['mean_tpot_ms']:.2f}ms")
        print(f"中位数TPOT: {metrics['median_tpot_ms']:.2f}ms")
        print(f"P99 TPOT: {metrics['p99_tpot_ms']:.2f}ms")
            
        return outputs

    def sync_run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests,
                        desc="Processed prompts",
                        dynamic_ncols=True)
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            # 这里计算input吞吐量和输出吞吐量
            for output in step_outputs:
                # 
                if output.finished:
                    if output.request_id in self.llm_engine.model_executor.driver_worker.model_runner.cpu_hack_kvcache_pool:
                        output.hack_kvs = self.llm_engine.model_executor.driver_worker.model_runner.cpu_hack_kvcache_pool[output.request_id]
                        self.llm_engine.model_executor.driver_worker.model_runner.cpu_hack_kvcache_pool.pop(output.request_id)
                    # 统计TTFT
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        return outputs
    
    def cacheblend_generate(self,
                            prompts: Optional[Union[str, List[str]]] = None,
                            sampling_params: Optional[Union[SamplingParams,
                                                            List[SamplingParams]]] = None,
                            prompt_token_ids: Optional[List[List[int]]] = None,
                            use_tqdm: bool = True,
                            ) -> List[RequestOutput]:
        """
        
        """
        # cache_fuse_metadata['collect'] = False
        # cache_fuse_metadata['check'] = False
        cache_fuse_metadata['recompute_mode'] = 0
        
        self.share_generate(prompts, sampling_params, prompt_token_ids, use_tqdm)
    
    def epic_generate(self,
                    prompts: Optional[Union[str, List[str]]] = None,
                    sampling_params: Optional[Union[SamplingParams,
                                                    List[SamplingParams]]] = None,
                    prompt_token_ids: Optional[List[List[int]]] = None,
                    use_tqdm: bool = True,
                                 ) -> List[RequestOutput]:
        """
        
        """
        # cache_fuse_metadata['collect'] = False
        # cache_fuse_metadata['check'] = False
        cache_fuse_metadata['recompute_mode'] = 2

        self.share_generate(prompts, sampling_params, prompt_token_ids, use_tqdm)
    
    def naive_generate(self,
                        prompts: Optional[Union[str, List[str]]] = None,
                        sampling_params: Optional[Union[SamplingParams,
                                                        List[SamplingParams]]] = None,
                        prompt_token_ids: Optional[List[List[int]]] = None,
                        use_tqdm: bool = True,
                        ) -> List[RequestOutput]:
        """
        
        """
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['check'] = False
        cache_fuse_metadata['recompute_mode'] = 3

        self.share_generate(prompts, sampling_params, prompt_token_ids, use_tqdm)

    def kvshare_generate(self,
                         prompts: Optional[Union[str, List[str]]] = None,
                         sampling_params: Optional[Union[SamplingParams,
                                                         List[SamplingParams]]] = None,
                         prompt_token_ids: Optional[List[List[int]]] = None,
                         use_tqdm: bool = True,
                         ) -> List[RequestOutput]:
        """
        
        """
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['check'] = False
        cache_fuse_metadata['recompute_mode'] = 1

        self.share_generate(prompts, sampling_params, prompt_token_ids, use_tqdm)


if __name__ == "__main__":
    
    llm = ShareLLM(model="/root/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c",
                    device="cuda:0",
                    dtype="bfloat16",
                    gpu_memory_utilization=0.7)
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    
    data_path = "/root/code/vllm_plus/examples/dataset/data/sharegpt/sharegpt90k_text_triplets.json"
    
    data = json.load(open(data_path))
    batch_size = 8
    
    # =================== 预计算 ======================
    # candiates = data["candidates"]

    # total_batches = (len(candiates) + batch_size - 1) // batch_size
    # for i in tqdm(range(0, len(candiates), batch_size), total=total_batches, desc="Processing batches"):
    #     batch_prompts = candiates[i:i + batch_size]
    #     llm.precompute_generate(batch_prompts, sampling_params=SamplingParams(max_tokens=1, temperature=0.0))
    
    # ================== 离线batch计算 ==================
    
    targets = data["targets"][:64]
    
    # total_batches = (len(targets) + batch_size - 1) // batch_size
    
    # for i in tqdm(range(0, len(targets), batch_size), total=total_batches, desc="Processing batches"):
    #     batch_prompts = targets[i:i + batch_size]
    #     llm.naive_generate(batch_prompts, sampling_params=SamplingParams(max_tokens=1, temperature=0.0))
    #     llm.sync_run_engine(use_tqdm=False)
    
    # targets = data["targets"]
    
    # 设置请求发送速率 (请求/s)
    rate = 8
    
    print(f"开始处理 {len(targets)} 个请求，速率为 {rate} 请求/s")
    
    # 直接将所有请求发送给_run_engine函数处理
    outputs = llm._run_engine(use_tqdm=True, rate=rate, requests=targets)
    
