import json
from pathlib import Path
from typing import Dict

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):

        # 参数校验
        chunks = self._step_1_validate_input(state)
        # 数据向量化
        output_data = self._step_2_generate_embeddings(chunks)
        # for item in output_data:
        #     item_name = item.get("item_name")
        #     print(f"item_name:{item_name}")
        #     sparse_vector = item["sparse_vector"]
        #     print(f"sparse_vector:{sparse_vector}")
        # 更新节点并返回结果
        state['chunks'] = output_data
        path = f"{Path(state.get('md_path')).parent}/{state.get('file_title')}_new_new_new_chunks.json" #修改三次后的（三个new）
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                output_data,
                f,
                ensure_ascii=False,
                indent=2
            )
        return state
    def _step_1_validate_input(self, state: ImportGraphState) -> list[Dict]:
        print("node_bge_embedding:步骤一：参数校验")
        # 参数校验
        chunks = state.get('chunks')
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)
        return chunks

    def _step_2_generate_embeddings(self, chunks: list[Dict[str,str]]) -> list[Dict[str,str]]:
        # 将item_name（块标题（商品名称））和content（块内容）转化为向量数据
        print("node_bge_embedding:步骤2：数据向量化")
        output_data = []   #处理后的文档
        batch_size = 5  # 批处理
        # 从x开始，走到y，每次走z步
        for i in range(0, len(chunks), batch_size):
            five_ready_texts = []   #把每个块中的ite_name和content提取出来并拼接，只对这两个字段进行向量化处理
            five_texts = chunks[i:i+batch_size]  #五个一组的快
            for doc in five_texts:
                item_name = doc['item_name']
                content = doc['content']
                five_ready_texts.append(f"{item_name}\n{content}" if item_name else content)
            embeddings = generate_embeddings(five_ready_texts)
            for j,doc in enumerate(five_texts):
                item = doc.copy()  #把最原来的拷贝过来（里边有五个字段）
                dense = embeddings["dense"][j]
                item["dense_vector"] = dense  #新增一个稠密向量字段
                sparse = embeddings["sparse"][j]
                item["sparse_vector"] = sparse  #新增一个稀疏向量字段
                output_data.append(item)


        return output_data

if __name__ == "__main__":
    node = NodeBGEEmbedding()
    with open(r"E:\AI Agent\g0525\output\H3C LA2608室内无线网关 用户手册-6W100-整本手册\H3C LA2608室内无线网关 用户手册-6W100-整本手册_new_chunks.json","r",encoding="utf-8") as f:
        chunks_content = f.read()
    # 把上一步读到的 JSON 字符串反序列化为 Python 对象（通常是 list[dict]，每个 dict 是一个 chunk）
    json_state = json.loads(chunks_content)
    init_state = {
        "chunks":json_state
    }
    response = node(init_state)
    # # 把节点返回的 response（处理后的 state 字典，里面 chunks 已带上向量）序列化成 JSON 字符串，用于打印查看
    # dumps = json.dumps(response,ensure_ascii=False,indent=4)
    # print(dumps)

