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
        self.ttft_recoders = {
            
        }
        self.throughput_recoders = {
            
        }
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
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['check'] = True
        
        # 在这里计算出全部的可以复用KV的位置
        batch_text_embedding = self.sentence_llm.encode(prompts)
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
        for idx, result in enumerate(results):
            if (len(result) > 0 and result[0]["distance"] >= 0.5 and cache_fuse_metadata['recompute_mode'] == 1) or \
                (len(result) > 0 and result[0]["distance"] >= 0.99 and cache_fuse_metadata['recompute_mode'] in [0,2]):
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
                    "kvcache": target_kvcache.to(self.llm_engine.model_executor.driver_worker.model_runner.device),
                    "reused_positions": [i for i in range(len(target_token_ids)) if i in target_matched_idx],
                    "unreused_positions": [i for i in range(len(target_token_ids)) if i not in target_matched_idx]
                }
            else:
                target_prompt = prompts[idx]
                target_token_ids = self.tokenizer.encode(target_prompt)
                # NOTE 注意修改变量
                target_kvcache = torch.zeros([32,2,len(target_token_ids),1024],device="cpu",dtype=torch.bfloat16)
                self.llm_engine.model_executor.driver_worker.model_runner.cpu_prefetch_kvcache_pool[current_request_id_start+idx] = {
                    "kvcache": target_kvcache.to(self.llm_engine.model_executor.driver_worker.model_runner.device),
                    "reused_positions": [],
                    "unreused_positions": [i for i in range(len(target_token_ids))]
                }
                

        self.generate(prompts, sampling_params, prompt_token_ids, use_tqdm, lora_request, multi_modal_data)

    
    
    def _run_engine(self, use_tqdm: bool, timeout: int = 60, request_queue: Optional[mp.Queue] = None) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests,
                        desc="处理请求进度",
                        dynamic_ncols=True)
        # Run the engine.
        outputs: List[RequestOutput] = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:  # 运行指定时间
            try:
                # 从队列获取请求，设置超时时间为0.1秒
                if request_queue is not None:
                    request = request_queue.get(timeout=0.1)
                    
                    if request is None:  # 收到结束标记
                        print("\n收到结束标记，引擎停止")
                        continue
                        
                    print("获取到请求")   
                    # 处理请求
                    self.generate(
                        prompts=[request['prompt']],
                        sampling_params=SamplingParams(max_tokens=512, temperature=0.0)
                    )
                
                # 处理引擎中的请求
                step_outputs = self.llm_engine.step()
                for output in step_outputs:
                    if output.finished:
                        if output.request_id in self.llm_engine.model_executor.driver_worker.model_runner.cpu_hack_kvcache_pool:
                            output.hack_kvs = self.llm_engine.model_executor.driver_worker.model_runner.cpu_hack_kvcache_pool[output.request_id]
                            self.llm_engine.model_executor.driver_worker.model_runner.cpu_hack_kvcache_pool.pop(output.request_id)
                        # 统计TTFT
                        self.ttft_recoders[output.request_id] = ( output.metrics.first_token_time - output.metrics.first_scheduled_time) * 1000
                        last_token_output_time = output.metrics.last_token_time
                        first_token_output_time = output.metrics.first_token_time
                        total_num_tokens = len(output.outputs[0].token_ids) + len(output.prompt_token_ids)
                        throughput = total_num_tokens / (last_token_output_time - first_token_output_time)
                        self.throughput_recoders[output.request_id] = throughput
                        
                        
                        outputs.append(output)
                        if use_tqdm:
                            pbar.update(1)
                            
            except mp.queues.Empty:
                # 如果队列为空，继续处理引擎中的请求
                continue
                
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
            for output in step_outputs:
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
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['check'] = False
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
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['check'] = False
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

def request_sender(queue, target_queue, rate):
    """
    子进程函数，按照指定间隔时间发送请求
    queue: 主进程的请求队列
    target_queue: 目标提示词队列
    rate: 请求间隔时间（秒）
    """
    start_time = time.time()
    request_count = 0
    total_requests = target_queue.qsize()  # 获取总请求数
    
    # 创建进度条
    pbar = tqdm(total=total_requests, desc="发送请求进度", dynamic_ncols=True)
    
    while time.time() - start_time < 60:  # 运行60秒
        try:
            # 从目标队列获取提示词，设置超时时间为0.1秒
            prompt = target_queue.get(timeout=0.1)
            
            # 向请求队列发送请求
            queue.put({
                'prompt': prompt,
                'timestamp': time.time()
            })
            request_count += 1
            pbar.update(1)  # 更新进度条
            
        except mp.queues.Empty:
            # 如果目标队列为空，结束进程
            print("\n目标队列为空，子进程结束")
            break
            
        time.sleep(rate)  # 等待指定的间隔时间
    
    # 关闭进度条
    pbar.close()
    
    # 发送结束标记
    queue.put(None)
    print(f"子进程完成，共发送 {request_count} 个请求")


if __name__ == "__main__":
    
    llm = ShareLLM(model="/root/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c",
                    device="cuda:0",
                    dtype="bfloat16",
                    gpu_memory_utilization=0.8)
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    
    data_path = "/root/code/vllm_plus/examples/dataset/data/sharegpt/sharegpt90k_text_triplets.json"
    
    data = json.load(open(data_path))
    batch_size = 8
    
    # =================== 预计算 ======================
    candiates = data["candidates"]
    # batch 32个一起提交
    
    total_batches = (len(candiates) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(candiates), batch_size), total=total_batches, desc="Processing batches"):
        batch_prompts = candiates[i:i + batch_size]
        llm.precompute_generate(batch_prompts, sampling_params=SamplingParams(max_tokens=1, temperature=0.0))
    
    # targets = data["targets"]
    
    # # =================== 测试 ======================
    # # 创建进程间通信队列
    # request_queue = mp.Queue()
    # target_queue = mp.Queue()

    # # 将目标提示词放入队列
    # for target in targets:
    #     target_queue.put(target)
    
    # # 创建并启动请求发送进程
    # rate = 0.1  # 每0.001秒发送1个请求
    # sender_process = mp.Process(
    #     target=request_sender,
    #     args=(request_queue, target_queue, rate)
    # )
    # sender_process.start()
    
    # # 启动引擎并等待停止事件
    # llm.engine_running = True
    
    # # 运行引擎
    # outputs = llm._run_engine(use_tqdm=True, timeout=60, request_queue=request_queue)
    
    # # 等待发送进程结束
    # sender_process.join()
 
    # # 打印统计信息
    # print("\n请求统计信息:")
    # print("TTFT统计:")
    # ttft_values = list(llm.ttft_recoders.values())
    # print(f"平均TTFT: {sum(ttft_values)/len(ttft_values):.2f}ms")
    # print(f"标准差TTFT: {np.std(ttft_values):.2f}ms")
    # print(f"最小TTFT: {min(ttft_values):.2f}ms")
    # print(f"最大TTFT: {max(ttft_values):.2f}ms")
    
    # throughput_values = list(llm.throughput_recoders.values())
    # print(f"平均吞吐量: {sum(throughput_values)/len(throughput_values):.2f} tokens/s")
    # print(f"标准差吞吐量: {np.std(throughput_values):.2f} tokens/s")
    # print(f"最小吞吐量: {min(throughput_values):.2f} tokens/s")
    # print(f"最大吞吐量: {max(throughput_values):.2f} tokens/s")
    