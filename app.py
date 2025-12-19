import os
import sys

# --- 1. 解决权限问题 ---
os.environ['GRADIO_TEMP_DIR'] = './gradio_temp'
if not os.path.exists('./gradio_temp'):
    os.makedirs('./gradio_temp')

import gradio as gr
import json
from main import ChatBot 

# --- 初始化 ChatBot ---
print("正在初始化系统，请稍候...")
try:
    bot = ChatBot()
except Exception as e:
    print(f"初始化失败: {e}")
    sys.exit(1)

def format_milvus_results(results):
    """格式化 Milvus 返回的原始对象为 JSON 友好格式"""
    formatted = []
    try:
        if not results: return []
        for hits in results:
            for hit in hits:
                item = {
                    "id": hit.id,
                    "score": round(hit.score, 4),
                    "text": hit.entity.get("text", "No text field")
                }
                formatted.append(item)
    except Exception as e:
        return {"error": f"解析错误: {str(e)}", "raw": str(results)}
    return formatted

def respond(message, chat_history):
    """
    Gradio 核心处理逻辑
    """
    if not message:
        return "", chat_history, None, ""
    
    # 防止 None
    if chat_history is None:
        chat_history = []

    try:
        # --- 调用 main.py 中的 chat 方法 ---
        # 接收：回答文本, 原始向量结果, 原始图谱结果
        answer, milvus_raw, graph_raw = bot.chat(message)

        # 1. 处理 Milvus 展示数据
        milvus_display = format_milvus_results(milvus_raw)

        # 2. 处理图谱展示数据
        if graph_raw:
            if isinstance(graph_raw, list):
                graph_display = "\n".join(graph_raw)
            else:
                graph_display = str(graph_raw)
        else:
            graph_display = "未找到相关图谱关联信息。"

    except Exception as e:
        answer = f"系统内部错误: {str(e)}"
        milvus_display = {"error": "Pipeline execution failed"}
        graph_display = str(e)

    # --- 修复点：使用字典格式 (Messages Format) ---
    # 根据报错信息 "Each message should be a dictionary with 'role' and 'content' keys"
    # 我们这里严格遵守该格式
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    
    return "", chat_history, milvus_display, graph_display

# --- 构建界面 ---

custom_css = """
.json-holder {max-height: 400px; overflow-y: scroll;}
.graph-holder {max-height: 400px; overflow-y: scroll;}
"""

with gr.Blocks(title="GraphRAG 知识库问答", css=custom_css) as demo:
    gr.Markdown("# 🕸️ GraphRAG: 向量+图谱混合检索问答系统")
    
    with gr.Row():
        # 左侧对话
        with gr.Column(scale=6):
            # 不传 type 参数，防止 TypeError，但喂给它字典数据
            chatbot = gr.Chatbot(
                label="对话窗口", 
                height=600, 
                avatar_images=(None, "🤖")
            )
            msg = gr.Textbox(label="输入问题", placeholder="例如：谁喜欢吃冰淇淋？", lines=2)
            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.ClearButton([msg, chatbot], value="清空")

        # 右侧信息
        with gr.Column(scale=4):
            gr.Markdown("### 🧠 思维链")
            with gr.Tabs():
                with gr.TabItem("Milvus 向量"):
                    milvus_output = gr.JSON(label="检索结果", elem_classes="json-holder")
                with gr.TabItem("Neo4j 图谱"):
                    graph_output = gr.TextArea(label="子图事实", lines=20, elem_classes="graph-holder")

    # 绑定事件
    msg.submit(respond, [msg, chatbot], [msg, chatbot, milvus_output, graph_output])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot, milvus_output, graph_output])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7111,
        share=False,
        allowed_paths=["./gradio_temp"]
    )