from config import settings
from prompt import CHAT_SYSTEM_PROMPT
from openai import OpenAI
from pymilvus import MilvusClient
from graph_retrieve import GraphRetriever
from FlagEmbedding import BGEM3FlagModel
from datetime import datetime
from zoneinfo import ZoneInfo 
import time



import torch

class ChatBot:
    def __init__(self, max_history=2):
        self.client = OpenAI(
            api_key=settings.CHAT_API_KEY,
            base_url=settings.CHAT_BASE_URL
        )
        self.system_prompt = CHAT_SYSTEM_PROMPT
        self.dialogue_history = []
        self.max_history = max_history
        
        self.milvus_client = MilvusClient(uri="./memory.db")
        print("milvus数据库连接成功！")
        self.embedding_model = BGEM3FlagModel(settings.EMBEDDING_MODEL_PATH, use_fp16=torch.cuda.is_available(), device="cuda" if torch.cuda.is_available() else "cpu")
        print("embedding模型加载成功！")
        self.neo4j_uri = settings.NEO4J_URI
        self.neo4j_auth = (settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        self.graph_retriever = GraphRetriever(milvus_client=self.milvus_client, collection_name=settings.MILVUS_COLLECTION, embedding_model=self.embedding_model, neo4j_uri=self.neo4j_uri, neo4j_auth=self.neo4j_auth)
        print("图数据库加载成功！")
        
    def maintain_history(self):
        if len(self.dialogue_history) >= self.max_history * 2:
            print(f"当前历史对话轮数为{len(self.dialogue_history)/2}，已删除最旧对话。")
            self.dialogue_history = self.dialogue_history[-(self.max_history-1)*2:]
            print(f"更新后历史对话轮数为{len(self.dialogue_history)/2}。")
        
    def chat(self, query):
        """
        核心对话逻辑
        Returns:
            tuple: (answer_str, milvus_results_raw, graph_context_list)
        """
        print("正在进行记忆检索...")
        # 1. 向量检索
        start_time = time.time()
        milvus_res = self.graph_retriever.base_retrieve(query, 3)
        print(f"向量检索耗时：{time.time() - start_time:.2f}秒")
        
        # 2. 图谱扩展
        start_time = time.time()
        memory_results = self.graph_retriever.expand_subgraph(milvus_res)
        print(f"检索到的记忆为：{memory_results}")
        print(f"图谱扩展耗时：{time.time() - start_time:.2f}秒")
        
        # 3. 构建时间
        beijing_time = datetime.now(ZoneInfo("Asia/Shanghai"))
        formatted_time = f"{beijing_time.year}年{beijing_time.month}月{beijing_time.day}日{beijing_time.hour:02d}:{beijing_time.minute:02d}"

        # 4. 构建 Prompt
        messages = [{"role": "system", "content": self.system_prompt.format(memory=memory_results, current_time=formatted_time)}] + self.dialogue_history
        messages.append({"role": "user", "content": query})
        
        # 5. 调用 LLM
        try:
            llm_res = self.client.chat.completions.create(
                model="Qwen-plus-ali",
                messages=messages
            )
            answer = llm_res.choices[0].message.content
        except Exception as e:
            answer = f"模型调用出错: {str(e)}"
        
        # 6. 更新历史
        self.dialogue_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer}
        ])
        self.maintain_history()
        
        # 返回 回答 以及 检索的中间过程数据
        return answer, milvus_res, memory_results
    
    
if __name__ == "__main__":
    
    chat_bot = ChatBot()
    while True:
        query = input("user：")
        answer, _, _  = chat_bot.chat(query)
        print(f"assistant：{answer}")
        
        
   

    