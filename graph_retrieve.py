from pymilvus import AnnSearchRequest, RRFRanker, MilvusClient
from FlagEmbedding import BGEM3FlagModel
import torch
from neo4j import GraphDatabase

class GraphRetriever:
    def __init__(self, milvus_client, collection_name, embedding_model, neo4j_uri, neo4j_auth):
        self.milvus_client = milvus_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    
    def close(self):
        self.driver.close()
    
    def text_embedding(self, text):
        if not text:
        # 返回空向量兜底
            return [0.0] * 1024, {}
            
        # return_dense=True, return_sparse=True, return_colbert_vecs=False
        output = self.embedding_model.encode(text, return_dense=True, return_sparse=True, return_colbert_vecs=False)

        # output['dense_vecs'] 是 numpy array，需要转 list
        dense_vec = output['dense_vecs'].tolist()

        # output['lexical_weights'] 是稀疏向量字典
        sparse_vec = output['lexical_weights']

        return dense_vec, sparse_vec 
    
    ## 基础检索
    def base_retrieve(self, query, k=3):
        query_dense, query_sparse = self.text_embedding(query)
        
        # 2. 构建稠密向量搜索请求
        req_dense = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"}, # 必须与建索引时的 metric_type 一致
            limit=3
        )

        # 3. 构建稀疏向量搜索请求
        req_sparse = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},     # 稀疏向量通常使用 IP (Inner Product)
            limit=3
        )

        # 4. 执行混合检索 (Hybrid Search)
        results = self.milvus_client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[req_dense, req_sparse],    # 传入两个请求
            ranker=RRFRanker(),              # 使用 RRF 算法进行结果融合重排序
            limit=k,
            output_fields=["text", "id"]
        )

        return results
    
    def expand_subgraph(self, milvus_results):
        retrieved_ids = []
        for hits in milvus_results:
            for hit in hits:
                retrieved_ids.append(hit['id'])
        
        retrieved_ids = list(set(retrieved_ids))
        contexts = [] 
        
        # 辅助函数：格式化属性字典为字符串
        def format_props(node_name, props):
            if not props: return None
            # 过滤掉非业务属性（根据你的实际情况增减）
            ignore_keys = ['id', 'label', 'dense_vector', 'sparse_vector', 'name', 'description']
            # 注意：我把 name 和 description 也过滤了，因为它们通常直接用于节点指代，
            # 剩下的才是需要补充的“详情属性” (比如 age, color, time 等)
            
            clean_props = {k: v for k, v in props.items() if k not in ignore_keys and v}
            if not clean_props: return None
            
            props_str = ", ".join([f"{k}: {v}" for k, v in clean_props.items()])
            return f"详情: [{node_name}] 的属性包含: {{{props_str}}}"
        
        # 辅助函数：获取节点显示名称 (优先 name, 其次 description, 最后 Unknown)
        def get_display_name(record, prefix):
            # prefix e.g., "start" or "end" or "center" or "neighbor"
            name = record.get(f'{prefix}_name')
            if not name:
                name = record.get(f'{prefix}_desc')
            return name if name else "未知节点"

        with self.driver.session() as session:
            for uid in retrieved_ids:
                # ==========================================
                # 策略 1: 处理节点 ID (Node Expansion)
                # ==========================================
                # 显式返回 n (中心点) 和 m (邻居点) 的各自信息
                node_cypher = """
                MATCH (n {id: $uid})-[r]-(m)
                RETURN 
                    // 1. 三元组构建 (保持真实方向)
                    startNode(r).name as start_node,
                    type(r) as relation_type,
                    r.name as relation_name,
                    endNode(r).name as end_node,
                    endNode(r).description as end_desc,
                    
                    // 2. 中心节点信息 (n)
                    n.name as center_name,
                    n.description as center_desc,
                    properties(n) as center_props,
                    
                    // 3. 邻居节点信息 (m)
                    m.name as neighbor_name,
                    m.description as neighbor_desc,
                    properties(m) as neighbor_props
                """
                
                node_res = session.run(node_cypher, uid=uid)
                is_node = False
                
                for record in node_res:
                    is_node = True
                    
                    # --- A. 构建事实三元组 ---
                    s = record['start_node'] if record['start_node'] else "未知"
                    r = record['relation_name'] if record['relation_name'] else record['relation_type']
                    # 终点可能是 Event，没有 name，取 description
                    e = record['end_node'] if record['end_node'] else record['end_desc']
                    if not e: e = "未知"
                    
                    contexts.append(f"事实: {s} {r} {e}")
                    
                    # --- B. 构建中心节点属性详情 ---
                    c_name = record['center_name'] if record['center_name'] else record['center_desc']
                    if c_name:
                        c_detail = format_props(c_name, record['center_props'])
                        if c_detail: contexts.append(c_detail)
                        
                    # --- C. 构建邻居节点属性详情 ---
                    n_name = record['neighbor_name'] if record['neighbor_name'] else record['neighbor_desc']
                    if n_name:
                        n_detail = format_props(n_name, record['neighbor_props'])
                        if n_detail: contexts.append(n_detail)

                # ==========================================
                # 策略 2: 处理关系 ID (Relation Expansion)
                # ==========================================
                if not is_node:
                    # 1. 先查这个 ID 对应的关系名字或类型
                    rel_name_cypher = "MATCH ()-[r {id: $uid}]->() RETURN r.name as name, type(r) as type LIMIT 1"
                    rel_check = session.run(rel_name_cypher, uid=uid).single()
                    
                    if rel_check:
                        target_name = rel_check['name']
                        target_type = rel_check['type']
                        
                        # 2. 查全图所有同类型关系，并返回头尾节点的 properties
                        if target_name:
                            # 匹配中文名
                            expand_all_rel_cypher = """
                            MATCH (a)-[r]->(b)
                            WHERE r.name = $target_name
                            RETURN 
                                a.name as start_name, 
                                a.description as start_desc,
                                properties(a) as start_props,  
                                
                                r.name as rel, 
                                
                                b.name as end_name, 
                                b.description as end_desc,
                                properties(b) as end_props     
                            """
                            params = {"target_name": target_name}
                        else:
                            # 匹配英文类型
                            expand_all_rel_cypher = """
                            MATCH (a)-[r]->(b)
                            WHERE type(r) = $target_type
                            RETURN 
                                a.name as start_name, 
                                a.description as start_desc,
                                properties(a) as start_props,  
                                
                                type(r) as rel, 
                                
                                b.name as end_name, 
                                b.description as end_desc,
                                properties(b) as end_props     
                            """
                            params = {"target_type": target_type}
                            
                        all_rels_res = session.run(expand_all_rel_cypher, **params)
                        
                        for record in all_rels_res:
                            # --- A. 构建事实 ---
                            s_name = get_display_name(record, 'start')
                            e_name = get_display_name(record, 'end')
                            r_name = record['rel']
                            
                            contexts.append(f"事实: {s_name} {r_name} {e_name}")
                            
                            # --- B. 构建属性详情 ---
                            s_props_str = format_props(s_name, record['start_props'])
                            if s_props_str: contexts.append(s_props_str)
                            
                            e_props_str = format_props(e_name, record['end_props'])
                            if e_props_str: contexts.append(e_props_str)
                            
        # 去重并返回
        return list(set(contexts))
    

    
    def retrieve_with_graph(self, query, k=3):
        # 1. 向量检索找到入口
        print(f"1. 正在检索 Milvus: {query}")
        milvus_res = self.base_retrieve(query, k)
        
        # 2. 图谱扩展获取上下文
        print(f"2. 正在扩展图谱子图...")
        graph_contexts = self.expand_subgraph(milvus_res)
        
        return graph_contexts
    
    
if __name__ == "__main__":
    
    milvus_client = MilvusClient(uri="./memory.db")
    collection_name = "memory_collection"
    local_model_path = "/home/qwen/yuzhipeng/models/bge-m3"
    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j", "12345678")
    embedding_model = BGEM3FlagModel(local_model_path, use_fp16=torch.cuda.is_available(), device="cuda" if torch.cuda.is_available() else "cpu")
    graph_retriever = GraphRetriever(milvus_clent=milvus_client, collection_name=collection_name, embedding_model=embedding_model, neo4j_uri=URI, neo4j_auth=AUTH)

    res = graph_retriever.retrieve_with_graph("谁喜欢吃冰淇淋？")
    print(res)