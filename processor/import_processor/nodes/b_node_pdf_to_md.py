import json
import logging
import time

from pathlib import Path

import requests

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, PdfConversionError
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
        print(f"已经获得下载url{zip_url}")

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
        """
        检验参数和配置信息
        """

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
        """
        上传并且获得下载链接
        """
        logging.info(f"_step_2_upload_and_poll上传到mineru服务器，并且获得下载地址")

        # 1.校验api_token和base_url
        api_token = mineru_config.api_token
        base_url = mineru_config.base_url
        if not api_token:
            raise FileProcessingError(message="api_token未配置")
        if not base_url:
            raise FileProcessingError(message="base_url未配置")

        # 2.申请上传来链接post
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        url = f"{base_url}/file-urls/batch"
        response = requests.post(url, headers=header, json=data)
        if response.status_code !=200:
            raise FileProcessingError(message=f"申请上传文件失败：{response.text}")
        result = response.json()
        if result.get("code") != 0:
            raise FileProcessingError(message=f"申请上传文件失败：{result.get('message')}")
        batch_id = result["data"]["batch_id"]
        signed_url = result["data"]["file_urls"][0]

        # 3.上传文件put
        with open(pdf_path_obj,"rb") as pdf_file:
            res_upload = requests.put(signed_url,data=pdf_file)
            if res_upload.status_code != 200:
                raise PdfConversionError(f"文件上传失败：状态码：{res_upload.status_code}，响应结果：{res_upload}")
            self.logger.info(f"文件上传成功！")

        # 4.获取下载链接get（循环+过期）
        poll_url = f"{base_url}/extract-results/batch/{batch_id}"  #检查转化结果
        start_time = time.time()  # 记录开始时间
        timeout_seconds = 600  # 最大超时时间
        poll_interval = 20  # 轮询间隔时间
        while True:
            if time.time() - start_time > timeout_seconds:
                raise PdfConversionError("【任务轮询】等待解析超时（超过10分钟）")

            try:
                # 核心：每次请求加上超时，并捕获网络异常
                res_poll = requests.get(url=poll_url, headers=header, timeout=15)  #获得下载链接
            except requests.exceptions.RequestException as e:
                logging.warning(f"【网络波动】请求异常：{e}，将在 {poll_interval} 秒后重试...")
                time.sleep(poll_interval)
                continue  # 网络异常就等3秒重来
            if res_poll.status_code != 200:
                raise PdfConversionError(f"【任务轮询】HTTP请求失败，状态码：{res_poll.status_code}，响应内容：{res_poll}")
            # 走到这行说明已经i请求成功了，但是不能证明转化任务成功了
            poll_data = res_poll.json()
            if poll_data.get("code") != 0:
                raise PdfConversionError(f"【任务轮询】业务错误，返回数据：{poll_data}")
            extract_results = poll_data["data"]["extract_result"]  #任务结果
            extract_result = extract_results[0]  #下载链接对象
            extract_state = extract_result["state"]  #下载状态
            print(f"转化结果：{extract_state}...")
            if extract_state == "done":
                full_zip_url = extract_result["full_zip_url"]
                return full_zip_url  #返回下载链接


    def _step_3_download_and_extract(self, zip_url, output_dir_obj, stem):
        logging.info(f"_step_3_download_and_extract下载并解压改名")
        # 1.下载
        response = requests.get(zip_url)
        if response.status_code != 200:
            raise FileProcessingError(message=f"获得下载文件失败:{response.text}")
        # 2.创建目录

        # 3.解压

        # 4.改名



if  __name__ == "__main__":
    setup_logging()
    init_state = {
        "pdf_path":r"E:\AI Agent\g0525\doc\H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf",
        "file_dir":r"E:\AI Agent\g0525\output"
    }
    node = NodePDFToMD()
    # result = node.process(init_state)不用写.process是因为NodePDFToMD继承了base.py而在base.py中有process方法可以自动调用
    result = node(init_state)
    dumps = json.dumps(result,ensure_ascii=False,indent=4)
    print(dumps)