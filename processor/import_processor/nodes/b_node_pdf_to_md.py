import json
import logging
from pathlib import Path

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行...")

        # 1.检查和获取相关参数
        pdf_path_obj,output_dir_obj = self._step_1_validate_paths(state)

        # 2.获取上传链接并上传文件到mineru服务器
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)

        # 3.下载zip压缩文件并解压改名
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        # 4.读取文档md_content，方便下一步处理图片
        with open(md_path,"r",encoding="utf-8") as md_file:
            md_content = md_file.read()
        # 5.设置state结果
        state["md_content"] = md_content
        state["md_path"] = md_path
        return state

    def _step_1_validate_paths(self, state:ImportGraphState):

        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path",expected_type=str)
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", expected_type=str)

        # 2.path路径封装
        # Path对象提供了一系列开箱即用的属性，无需自己写正则或字符串切割去提取文件名：
        # .stem：获取不带后缀的文件名（如AolynkCB304n）
        # .name：获取带后缀的文件名（如olynkCB304n.pdf）
        # .suffix：获取后缀（如.pdf）
        # .parent：获取父目录
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(file_dir)


        # 3.文档是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在：{pdf_path}")
        if not output_dir_obj.exists():
            raise FileProcessingError(message=f"输出目录不存在：{output_dir_obj}")
        return pdf_path_obj,output_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj):
        logging.info(f"_step_2_upload_and_poll上传到mineru服务器...")
        return "下载url"

    def _step_3_download_and_extract(self, zip_url, output_dir_obj, stem):
        logging.info(f"_step_3_download_and_extract下载并解压改名")
        return "md_path"

if  __name__ == "__main__":
    setup_logging()
    init_state = {
        "pdf_path":"E:\AI Agent\g0525\doc\hak180产品安全手册.pdf",
        "file_dir":"E:\AI Agent\g0525\output"
    }
    node = NodePDFToMD()
    # result = node.process(init_state)不用写.process是因为NodePDFToMD继承了base.py而在base.py中有process方法可以自动调用
    result = node(init_state)
    dumps = json.dumps(result,ensure_ascii=False,indent=4)
    print(dumps)