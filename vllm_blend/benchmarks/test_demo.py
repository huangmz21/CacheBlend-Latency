import torch
from collections import deque
from typing import List
from vllm import LLMEngine, SamplingParams, LLM
from typing import List, Optional, Union
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import MultiModalData
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import MilvusClient
import time
import os
from edit2 import TokenMatch, match_sequences
import uuid
os.environ["VLLM_USE_MODELSCOPE"] = "False"

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
        
        self.tokenizer = self.get_tokenizer()
    
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
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['check'] = False
        return super().generate(prompts, sampling_params, prompt_token_ids, use_tqdm, lora_request, multi_modal_data)
    
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
        output = super().generate(prompts, sampling_params, prompt_token_ids, use_tqdm, lora_request, multi_modal_data)
        # 将生成的KV Cache保存到Milvus
        batch_text_embedding = self.sentence_llm.encode(prompts)
        for idx, output in enumerate(output):
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
        return output
    
    
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
                
        # =====================结束HACK
        outputs = super().generate(prompts, sampling_params, prompt_token_ids, use_tqdm, lora_request, multi_modal_data)
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['check'] = False
    
        return outputs
    
    
    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
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
                    output.metrics.first_scheduled_time = self.llm_engine.model_executor.driver_worker.model_runner.real_schedule_time[output.request_id]
                    output.metrics.first_token_time = self.llm_engine.model_executor.driver_worker.model_runner.real_first_token_time[output.request_id]
                    outputs.append(output)
                    
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
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
        
        return self.share_generate(prompts, sampling_params, prompt_token_ids, use_tqdm)
    
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

        return self.share_generate(prompts, sampling_params, prompt_token_ids, use_tqdm)
    
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

        return self.share_generate(prompts, sampling_params, prompt_token_ids, use_tqdm)

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

        return self.share_generate(prompts, sampling_params, prompt_token_ids, use_tqdm)


if __name__ == "__main__":
    
    llm = ShareLLM(model="/root/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c",device="cuda:0",dtype="bfloat16")
    
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    
    candidate_prompts = ["My name is Huan Yang, I come from China. What is your name? ","This is China, I come from China. What is your name? "]
    target_prompts = ["My name is Huan Yang, I come from China. What is your name? " * 50, "This is China, I come from China. What is your name? " * 50]
    
    short_sampling_params = [SamplingParams(temperature=0.0,max_tokens=1),SamplingParams(temperature=0.0,max_tokens=1)]
    long_sampling_params = [SamplingParams(temperature=0.0,max_tokens=128),SamplingParams(temperature=0.0,max_tokens=128)]
    
    outputs = llm.precompute_generate(prompts=candidate_prompts,
                                sampling_params=short_sampling_params)
    
    print("================ Cacheblend ================")
    avg_ttft = []
    for _ in range(3):
        outputs = llm.cacheblend_generate(prompts=target_prompts,use_tqdm=False,
                                    sampling_params=long_sampling_params)
        
        
        for output in outputs:
            ttft = output.metrics.first_token_time - output.metrics.first_scheduled_time
            # print(output.outputs[0].text)
            # print(ttft)
            avg_ttft.append(ttft)
    print(avg_ttft)
    print(sum(avg_ttft)/len(avg_ttft))
        
    print("================ KVShare ================")
    avg_ttft = []
    for _ in range(5):
        outputs = llm.kvshare_generate(prompts=target_prompts,use_tqdm=False,
                                    sampling_params=long_sampling_params)
        
        for output in outputs:
            ttft = output.metrics.first_token_time - output.metrics.first_scheduled_time
            avg_ttft.append(ttft)
    print(avg_ttft)
    print(sum(avg_ttft)/len(avg_ttft))


    print("================ Naive ================")
    avg_ttft = []
    for _ in range(3):
        outputs = llm.naive_generate(prompts=target_prompts,use_tqdm=False,
                                    sampling_params=long_sampling_params)
        
        for output in outputs:
            ttft = output.metrics.first_token_time - output.metrics.first_scheduled_time
            avg_ttft.append(ttft)
    print(sum(avg_ttft)/len(avg_ttft))
    print("================ Full Compute ================")
    avg_ttft = []
    for _ in range(3):
        outputs = llm.generate(prompts=target_prompts,use_tqdm=False,
                                sampling_params=long_sampling_params)
        
        for output in outputs:
            ttft = output.metrics.first_token_time - output.metrics.first_scheduled_time
            avg_ttft.append(ttft)
    print(avg_ttft)
    print(sum(avg_ttft)/len(avg_ttft))
