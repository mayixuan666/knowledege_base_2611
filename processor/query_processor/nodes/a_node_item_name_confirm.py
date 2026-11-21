import json
from typing import Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, \
    ITEM_NAME_EXTRACT_TEMPLATE
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.llm_utils import get_llm_client

from utils.mongo_history_util import get_recent_messages,save_chat_message


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # 1.参数校验
        session_id, original_query = self._step_1_validate_param(state)
        logger.info(f"步骤1：参数校验通过")

        # 2.获取历史记忆
        history = get_recent_messages(session_id)
        logger.info(f"步骤2：获取到 {len(history)} 条历史消息")
        state["history"] = history

        # 3.保存用户信息(存原始问题)
        message_id = save_chat_message(session_id, "user", original_query)
        logger.info(f"步骤3：用户消息已初始保存, ID: {message_id}")

        # 4.LLM提取:rewritten_query（改写问题）、item_name（模型识别到的设备主体）
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res.get("item_names")
        rewritten_query = extract_res.get("rewritten_query", original_query)
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names
        print("extract_res:", extract_res)

        # 5.向量搜索（搜索知识库）、6.搜索结果对齐（整理）
        align_result = {}
        if len(item_names) >= 0:
            query_results = self._step_5_vectorize_and_query(item_names)
            align_result = self._step_6_align_item_names(query_results)  #返回一个相似度最高的结果
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        # 7.状态state状态整理（更新state数据），将第六步结果记录到state
        state = self._step_7_check_confirmation(state, align_result, history)

        # 8.保存历史会话(将识别完的问题保存到历史记录)
        self._step_8_write_history(state, session_id, rewritten_query, message_id)
        return state

    def _step_1_validate_param(self, state: QueryGraphState) -> Tuple[str, str]:

        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心参数original_query缺失")

        return session_id, original_query

    def _step_4_extract_info(self, original_query, history) -> Dict:
        print("step4:模型提取意图")

        ai_client = get_llm_client()
        # 拼接上下文：history+original_query
        history_text = ""
        for msg in history:
            role = msg.get("role")
            content = msg.get("text")
            history_text += f"{role}: {content}\n"
        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text,
            query=original_query
        )
        messages = [
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = ai_client.invoke(messages)
        response_content = response.content
        result = json.loads(response_content)
        if "item_names" not in result:
            result["item_names"] = []
            # 确保返回结果包含rewritten_query字段，无则复用原始查询
        if "rewritten_query" not in result:
            result["rewritten_query"] = original_query
        if result["item_names"]:
            result["item_names"] = [
                name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                for name in result["item_names"]
            ]


        return result

    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        print("step5:向量化并检索，返回与item_name对应的向量库中相似的名称列表")

        result: List[Dict] = []

        return result

    def _step_6_align_item_names(self, query_results) -> dict:
        print("step6:对齐结果，高份结果低份结果整合")
        return{
            "confirm_item_name":[],  # 确认后商品名称（>0.8分）
            "options":[]     # 低分商品名称(>0.6)


        }

    def _step_7_check_confirmation(self, state, align_result:Dict, history):
        print("step7:状态state信息，根据第六步高分低分对齐结果整理")
        return state

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):

        print("step8:写入历史会话，更新")





if __name__ == "__main__":

    # 初始化图状态
    init_state = {
        "original_query": "H3C这个玩应咋搞的啊？",
        "session_id":123
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    # json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logger.info(result)