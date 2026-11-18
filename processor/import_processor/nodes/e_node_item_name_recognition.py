from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取,对当前文档的总结实体名称
    """


    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):


        return state