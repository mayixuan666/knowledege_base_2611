from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState

#对主题设备名称精准总结识别，不是总结整个文档
class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):


        return state