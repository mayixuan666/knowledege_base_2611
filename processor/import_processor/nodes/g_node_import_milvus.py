import json
import logging
from typing import Dict, Any, List

from pymilvus import DataType, MilvusClient
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode,setup_logging
from processor.import_processor.exceptions import StateFieldError, MilvusError
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import get_milvus_client,escape_milvus_string

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.state import ImportGraphState

#对主题设备名称精准总结识别，不是总结整个文档
class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):

        # 1.数据校验
        chunks_json_data, vector_dimension = self._step_1_check_inputs(state)
        # 2.客户端和集合准备（字段+索引）
        milvus_client = self._step_2_prepare_collection(vector_dimension)
        # 3.清理可能的冗余数据（幂等性）
        self._step_3_clean_old_data(milvus_client, chunks_json_data)
        # 4.数据入库，返回数据库主键
        update_chunks = self._step_4_insert_data(milvus_client, chunks_json_data)
        for chunk in update_chunks:
            print(f"chunk_id:{chunk['chunk_id']}")
        # 更新状态
        state["chunks"] = update_chunks
        return state

    def _step_1_check_inputs(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        # 处理文本块
        chunks_json_name = state.get("chunks")
        if not chunks_json_name:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        # 提取向量维度
        dimension = len(chunks_json_name[0]['dense_vector'])
        print(f"向量维度：{dimension}")
        return chunks_json_name, dimension

    def _step_2_prepare_collection(self, vector_dimension: int):
        print("node_import_milvus:_step_2_prepare_collection")
        # 1.客户端
        milvus_client = get_milvus_client()
        # 2.集合
        collection_name = milvus_config.chunks_collection
        if not milvus_client.has_collection(collection_name):
            self.create_chunks_collection(milvus_client,collection_name, vector_dimension)

        return milvus_client

    def _step_3_clean_old_data(self, client, chunks_json_data):
        print("node_import_milvus:_step_3_clean_old_data")
        # 1. 获取查询条件
        file_title = chunks_json_data[0].get("file_title")

        # 2. 执行幂等清理
        self.clear_chunks_by_file_title(client, file_title)
    # 批量插入milvus
    def _step_4_insert_data(self, milvus_client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("node_import_milvus:_step_4_insert_data")
        # 数据处理
        data_to_insert = []
        for item in chunks_json_data:
            item_copy = item.copy()
            if "part" not in item_copy:
                item_copy["part"] = 0
            data_to_insert.append(item_copy)
        # 批量插入
        insert_result = milvus_client.insert(collection_name=milvus_config.chunks_collection, data=data_to_insert)
        insert_count = insert_result.get("inserted_count")

        # 回填主键，将数据插入数据库中时生成的主键回填到state中
        insert_ids = insert_result.get("ids")
        for index,item in enumerate(chunks_json_data):
            item["chunk_id"] = str(insert_ids[index])
        # 返回带有主键的chunks
        return chunks_json_data
    # 步骤2方法1
    def create_chunks_collection(self,milvus_client,collection_name,vector_dimension):
        print("node_import_milvus:create_chunks_collection")
        # 1. 创建schem
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 2. 字段声明
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片内容
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)  # 切片标题
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100)  # 父标题
        schema.add_field(field_name="part", datatype=DataType.INT8)  # 分片编号
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)  # 源文件标题
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)  # 商品名称（幂等性依据）
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # 稠密向量

        # 3. 索引声明
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        # 创建集合
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

    # 步骤3方法1
    def clear_chunks_by_file_title(self, client, file_title):

        try:
            file_title = escape_milvus_string(file_title)
            client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=f"file_title=='{file_title}'")
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败: {str(e)}")
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}")

if __name__ == "__main__":

    # setup_logging()

    json_path = r"E:\AI Agent\g0525\output\H3C LA2608室内无线网关 用户手册-6W100-整本手册\H3C LA2608室内无线网关 用户手册-6W100-整本手册_new_new_new_chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state
    }

    # 执行核心处理流程
    node_import_milvus = NodeImportMilvus()
    result = node_import_milvus(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))



