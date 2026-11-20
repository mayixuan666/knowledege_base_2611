import json
from email import message
from typing import List, Dict, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph_sdk import schema
from mmengine.evaluator import metric
from pymilvus import DataType
from pymongo import collection

from config.lm_config import lm_config
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import get_milvus_client
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import escape_milvus_string


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取,对当前文档的总结实体名称
    """


    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):

        # 1.参数处理
        file_title, chunks = self._step_1_get_inputs(state)

        # 2.上下文拼接
        context = self._step_2_build_context(file_title,chunks)
        print(context)

        # 3.模式识别（总结）
        item_name = self._step_3_call_llm(file_title, context)
        print(f"item_name:{item_name}")
        # 4.回填数据（item_name -> chunks）
        self._step_4_update_chunks(state,chunks,item_name)

        # 5.主体名向量化(返回稠密稀疏向量)
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)
        print(f"dense_vector:{dense_vector}")
        print(f"sparse_vector:{sparse_vector}")

        # 6.存入milvus向量库
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        return state

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, List[Dict]]:

        print("node_item_name_recognition:步骤1：参数处理")
        file_title = state['file_title']

        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state['chunks']

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return file_title, chunks

    def _step_2_build_context(self, file_title,chunks:List[Dict]) -> str:

        print("node_item_name_recognition:步骤2：上下文拼接")
        # 限制上下文片数
        k = self.config.item_name_chunk_k
        # 限制切片长度
        chunk_size = self.config.item_name_chunk_size
        parts:List[str] = []
        total_chars = 0
        for index,chunk in enumerate(chunks[:k],start=1):
            chunk_title = chunk.get('title','').strip()
            chunk_content = chunk.get('content','').strip()

            # 格式化
            piece = f"【切片{index}】\n标题{chunk_title}\n内容：{chunk_content}"
            parts.append(piece)

            total_chars +=len(piece)

            if(total_chars > chunk_size):
                break

            # 截断处理
            context = "\n\n".join(parts).strip()

            # 对返回结果进行二次截断
            final_context = context[:chunk_size]

        return final_context

    def _step_3_call_llm(self, file_title, context) -> str:
        print("node_item_name_recognition:步骤3：模式识别")
        if not file_title:
            return file_title

        # llm
        llm_ai = ChatOpenAI(
            model=lm_config.llm_model,
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            temperature=lm_config.llm_temperature,
            extra_body={"enable_thinking": False},
        )
        # 提示词
        prompt = f"""
            请从以下信息中识别出商品名称与型号：
            文件名：{file_title}
                
            正文切片（用于辅助识别）：
            {context}
            
            要求：
            1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
            2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
            3. 如果无法识别商品名称,请返回空字符串。
        """
        message = [
            SystemMessage("你是一个专业的商品名称识别模型，请根据提供的信息，识别商品名称。名称最好不要超过20字"),
            HumanMessage(content=prompt)
        ]
        # 调用
        response = llm_ai.invoke(message)
        # 解析，数据清洗
        item_name = getattr(response,"content","").strip()  #主体名称
        item_name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")

        # 兜底
        if not item_name:
            return file_title
        return item_name

    def _step_4_update_chunks(self, state,chunks,item_name)  :
        print("node_item_name_recognition:步骤4：回填数据")
        state["item_name"] = item_name
        for chunk in chunks:
            chunk["item_name"] = item_name
        return state

    def _step_5_generate_vectors(self, item_name) :
        print("node_item_name_recognition:步骤5：返回稠密稀疏数据")
        embeddings = generate_embeddings([item_name])
        dense = embeddings['dense'][0]
        sparse = embeddings['sparse'][0]
        return dense, sparse


    def _step_6_save_to_milvus(self, state, file_title, item_name, dense_vector, sparse_vector):
        print("node_item_name_recognition:步骤6：存入milvus向量库")
        milvus_url = milvus_config.milvus_url  # 链接地址
        collection_name = milvus_config.item_name_collection  # 表名
        if not milvus_url or not collection_name:
            raise Exception("Milvus配置错误")
        client = get_milvus_client()

        # 字段
        schema = client.create_schema(auto_id=True,enable_dynamic_field=True)
        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=100
        )
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=100
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        # 索引
        index_params = client.prepare_index_params()
        # 为稠密向量创建索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",  # 分组+精确搜索
            metric_type="COSINE",
            params={"nlist":128}  # 聚类数（影响检索精度/速度）
        )
        # 为稀疏向量创建索引
        index_params.add_index(
            field_name="sparse_vector",  # 字段名
            index_name="sparse_vector_index",  # 索引名
            index_type="SPARSE_INVERTED_INDEX",  # 索引类型
            metric_type="IP",  # 相似度计算方式（内积）
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                # 高效的稀疏检索算法

                "normalize": True,
                # ↑ L2 归一化，让内积 (IP) 等价于余弦相似度

                "quantization": "none"
                # ↑ 关闭量化，保持原始精度：模型生成的向量已经压缩的一半的精度了（BGE_FP16=1），这里就不再压缩了
                # "quantization": "none" → 存储原始向量，不压缩
                # "quantization": "sq8" → 存储压缩后的向量（8-bit 量化
            })

        # 建表
        collection_name = milvus_config.item_name_collection
        if not client.has_collection(collection_name):
            client.create_collection(collection_name=collection_name,schema=schema,index_params=index_params)

        # 3. 幂等性处理：清理同名商品数据（避免重复存储）
        # 转义商品名称（防止特殊字符导致filter解析失败）
        safe_item_name = escape_milvus_string(item_name)
        # 构建过滤表达式：item_name等于目标值
        filter_expr = f'item_name=="{safe_item_name}"'
        # 删除符合条件的数据
        client.delete(collection_name=collection_name, filter=filter_expr)


        # 插入
        data = {
            "file_title": file_title,
            "item_name": item_name,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        }
        client.insert(collection_name,data)
        client.load_collection(collection_name)  # 将表从存储引擎加载到搜索引擎，为了将来查相似度
        state["item_name"] = item_name

if __name__ == '__main__':
    node = NodeItemNameRecognition()
    path = r"E:\AI Agent\g0525\output\H3C LA2608室内无线网关 用户手册-6W100-整本手册\H3C LA2608室内无线网关 用户手册-6W100-整本手册_new_chunks.json"
    with open(path,"r",encoding="utf-8") as f:
        chunks_json_data = f.read()

    init_state = {
        "file_title":"H3C LA2608室内无线网关 用户手册-6W100-整本手册",
        "chunks":json.loads(chunks_json_data)
    }
    process = node.process(init_state)
    print(process)
