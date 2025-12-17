import pandas as pd
from neo4j import GraphDatabase
import os
import uuid

# ================= 配置 =================
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "12345678")

# 输入文件路径 (请根据实际情况确认路径)
NODES_FILE = "/home/qwen/yuzhipeng/memory/chatbot_memory/data/nodes.csv"
RELS_FILE = "/home/qwen/yuzhipeng/memory/chatbot_memory/data/relations.csv"

# 输出文件路径
OUTPUT_NODES_UUID = "/home/qwen/yuzhipeng/memory/chatbot_memory/data/nodes_with_uuid.csv"
OUTPUT_RELS_UUID = "/home/qwen/yuzhipeng/memory/chatbot_memory/data/relations_with_uuid.csv"

class SimpleGraphImporter:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Neo4j 数据库已清空。")

    def import_nodes(self, df):
        with self.driver.session() as session:
            for _, row in df.iterrows():
                # 排除不需要放入 props 的字段
                exclude_keys = ['id', 'label'] 
                props = {k: v for k, v in row.items() if pd.notna(v) and k not in exclude_keys}
                
                # id 显式设置，其他放入 props
                cypher = f"MERGE (n:{row['label']} {{id: $id}}) SET n += $props"
                session.run(cypher, id=row['id'], props=props)
        print(f"节点导入完成，共 {len(df)} 条。")

    def import_relationships(self, df):
        with self.driver.session() as session:
            for _, row in df.iterrows():
                rel_type = row['relation_type']
                
                # 排除建立关系所用的 ID 字段以及我们新增的主键 id
                exclude_keys = ['source_id', 'target_id', 'relation_type', 'id']
                props = {k: v for k, v in row.items() if pd.notna(v) and k not in exclude_keys}

                # 创建关系并设置 id 和其他属性
                # 注意：这里使用 MERGE 仅仅基于方向和类型，然后 SET id
                # 如果同一对节点间有多个相同类型的关系，可能需要更复杂的 MERGE 策略，
                # 但由于前面 clear_database 了，这里每次都是新建。
                cypher = f"""
                MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.id = $rel_id
                SET r += $props
                """
                session.run(cypher, 
                            source_id=row['source_id'], 
                            target_id=row['target_id'], 
                            rel_id=row['id'],  # 传入关系的 UUID
                            props=props)
        print(f"关系导入完成，共 {len(df)} 条。")

# ================= 数据预处理函数 =================
def process_data_with_uuid(nodes_df, rels_df):
    print("正在生成 UUID 并替换原有 ID...")
    
    # 1. 建立节点映射字典: { 旧ID : 新UUID }
    old_ids = nodes_df['id'].unique()
    id_map = {old_id: str(uuid.uuid4()) for old_id in old_ids}
    
    # 2. 替换节点表 ID
    nodes_df['id'] = nodes_df['id'].map(id_map)
    
    # 3. 替换关系表 source_id 和 target_id (使用节点的 UUID)
    rels_df['source_id'] = rels_df['source_id'].map(id_map)
    rels_df['target_id'] = rels_df['target_id'].map(id_map)
    
    # 4. 【新增】为关系表生成独立的 UUID，字段名为 'id'
    # 每一行关系数据生成一个唯一的 UUID
    rels_df['id'] = [str(uuid.uuid4()) for _ in range(len(rels_df))]
    
    # 调整列顺序，把 id 放在第一列好看一点（可选）
    cols = ['id'] + [c for c in rels_df.columns if c != 'id']
    rels_df = rels_df[cols]
    
    return nodes_df, rels_df

if __name__ == "__main__":
    # 1. 检查文件
    if not os.path.exists(NODES_FILE) or not os.path.exists(RELS_FILE):
        print(f"错误：文件不存在。\n{NODES_FILE}\n{RELS_FILE}")
        exit(1)

    try:
        # 2. 读取原始 CSV
        print("读取 CSV 文件...")
        nodes_df = pd.read_csv(NODES_FILE)
        rels_df = pd.read_csv(RELS_FILE)
        
        # 3. 处理 UUID (节点ID替换 + 关系ID新增)
        nodes_df, rels_df = process_data_with_uuid(nodes_df, rels_df)
        
        # 4. 保存处理后的文件
        print(f"保存带 UUID 的新数据到: {OUTPUT_NODES_UUID}")
        print(f"保存带 UUID 的新数据到: {OUTPUT_RELS_UUID}")
        nodes_df.to_csv(OUTPUT_NODES_UUID, index=False)
        rels_df.to_csv(OUTPUT_RELS_UUID, index=False)
        
        # 5. 导入 Neo4j
        print("开始导入 Neo4j...")
        importer = SimpleGraphImporter(URI, AUTH)
        try:
            importer.clear_database()
            importer.import_nodes(nodes_df)
            importer.import_relationships(rels_df)
            print("✅ 所有操作成功完成！")
        finally:
            importer.close()
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()