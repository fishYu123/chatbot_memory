from pymilvus import MilvusClient, DataType
from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import torch

## 加载embedding模型
local_model_path = "/home/qwen/yuzhipeng/models/bge-m3"
model = BGEM3FlagModel(local_model_path, use_fp16=torch.cuda.is_available(), device="cuda" if torch.cuda.is_available() else "cpu")
print("embedding 模型加载完成..")

def get_embeddings(text):
    if not text:
        # 返回空向量兜底
        return [0.0] * 1024, {}
    
    # return_dense=True, return_sparse=True, return_colbert_vecs=False
    output = model.encode(text, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    
    # output['dense_vecs'] 是 numpy array，需要转 list
    dense_vec = output['dense_vecs'].tolist()
    
    # output['lexical_weights'] 是稀疏向量字典
    sparse_vec = output['lexical_weights']
    
    return dense_vec, sparse_vec


milvus_client = MilvusClient(uri="./memory.db")
collection_name = "memory_collection"

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)
    print(f"检测到已存在集合，正在删除原有集合{collection_name}")   
    
## 定义schema
schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=True)


schema.add_field(
    field_name="id",
    datatype=DataType.VARCHAR,
    max_length=50,
    is_primary=True,
)
schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    max_length=20
)
schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

milvus_client.create_collection(collection_name=collection_name, schema=schema)

## 索引建立
index_params =milvus_client.prepare_index_params()

index_params.add_index(
    field_name="dense_vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP"
),

index_params.add_index(
    field_name="id",
    index_type="INVERTED"
)
index_params.add_index(
    field_name="text",
    index_type="INVERTED"
)

milvus_client.create_index(
    collection_name=collection_name,
    index_params=index_params
)


## 插入数据

nodes = pd.read_csv("data/nodes_with_uuid.csv")
relations = pd.read_csv("data/relations_with_uuid.csv")

insert_data = []

print("正在处理节点数据...")
for _, row in nodes.iterrows():
    uuid = row['id']
    label = row['label']
    
    # 逻辑判断：不同类型的节点，取不同的字段作为 Embedding 对象
    content_text = ""
    
    if label in ["Person", "Item"]:
        # 人和物品取 'name'
        if pd.notna(row['name']):
            content_text = str(row['name'])
            
    elif label == "Event":
        # 事件取 'description'
        if pd.notna(row['description']):
            content_text = str(row['description'])
            
    # 只有当文本不为空时才存入 Milvus
    if content_text:
        dense, sparse = get_embeddings(content_text)
        insert_data.append({
            "id": uuid,
            "text": content_text,
            "dense_vector": dense,
            "sparse_vector": sparse
        })

print("正在处理关系数据...")
relation_dueplicate = {
    
}
for _, row in relations.iterrows():
    uuid = row['id']
    
    relation_name = row['name']
    if relation_name not in relation_dueplicate:
        relation_dueplicate[relation_name] = uuid
    else:
        continue
    
    
    # 关系取 'name' (例如: 喜欢, 爸爸, 参与)
    if pd.notna(row['name']):
        content_text = str(row['name'])
        dense, sparse = get_embeddings(content_text)
        
        insert_data.append({
            "id": uuid,
            "text": content_text,
            "dense_vector": dense,
            "sparse_vector": sparse
        })

# ================= 6. 执行插入 =================
print(f"准备插入 {len(insert_data)} 条数据...")

if insert_data:
    # 批量插入
    batch_size = 100
    for i in range(0, len(insert_data), batch_size):
        batch = insert_data[i : i + batch_size]
        milvus_client.insert(collection_name=collection_name, data=batch)
        print(f"已插入: {min(i + batch_size, len(insert_data))} / {len(insert_data)}")
        
    print("✅ 数据插入成功！")
else:
    print("⚠️ 没有有效数据被插入（请检查CSV是否为空或字段名是否正确）。")

# ================= 7. 简单验证 =================
# print("\n--- 检索测试: '谁喜欢冰淇淋?' ---")
# query = "谁喜欢冰淇淋"
# q_dense, _ = get_embeddings(query)

# results = milvus_client.search(
#     collection_name=collection_name,
#     data=[q_dense],
#     anns_field="dense_vector",
#     limit=3,
#     output_fields=["text", "id"]
# )

# for hits in results:
#     for hit in hits:
#         print(f"Found ID: {hit['entity']['id']} | Text: {hit['entity']['text']} | Dist: {hit['distance']:.4f}")