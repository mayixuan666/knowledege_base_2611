import base64
import json
import logging
import os
import re
import time
from collections import deque

from pathlib import Path
from typing import Tuple, List, Dict,Deque
from urllib import request

from minio import Minio
from minio.deleteobjects import DeleteObject

from config.minio_config import minio_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.state import ImportGraphState
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from utils.llm_utils import get_llm_client
from utils.minio_utils import get_minio_client
from config.lm_config import lm_config



class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行...")


        # 1.参数处理
        md_content,md_path_obj,images_dir = self._step_1_get_content(state)
        print(f"md_content:{md_content}, md_path_obj:{md_path_obj}, images_dir:{images_dir}")

        # 2.图片处理
        # 需要知道被扫描图片在md文档中的位置，故需要md_content这个参数
        target_images = self._step_2_scan_images(md_content,images_dir)
        print(f"target_images:{target_images}")

        # 3.视觉模型摘要
        summaries = self._step_3_generate_summaries(md_path_obj.stem,target_images)
        print(f"summaries:{summaries}")

        # 4.上传minio,替换md
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)

        # 5.保存备份md
        new_md_file_name = self._step_5_backup_new_md_file(state['md_path'], new_md_content)

        # # 6.更新state
        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name

        return state

    def _step_1_get_content(self, state):
        # 1.校验参数
        md_path = state['md_path']
        if not md_path:
            raise StateFieldError(field_name='md_path', expected_type=str)
        md_path_obj = Path(md_path)
        md_content = state['md_content']
        if not md_content:
            with open(md_path,"r",encoding="utf-8") as md_file:
                md_content = md_file.read()
        # 2.图片路径,和md文件同路径的images下的图片
        images_dir = md_path_obj.parent / 'images'

        return md_content,md_path_obj,images_dir

    def _step_2_scan_images(self, md_content,images_dir):

        # 1.返回结果
        target_images = []

        # 2.扫描图片
        for image_file in os.listdir(images_dir):
            file_ext = os.path.splitext(image_file)[1].lower()

            if file_ext not in self.config.image_extensions:
                self.logger.warning(f"图片格式不支持，跳过：{image_file}")
                continue
            # 图片路径
            img_path = images_dir / image_file
            # 找到图片上下文
            context = self._find_image_in_md(md_content,image_file)
            # 过滤MD中未引用的图片
            if not context:
                self.logger.warning(f"图片未在MD中引用，跳过处理：{image_file}")
                continue
            target_images.append((image_file,img_path,context))

        return target_images
    # 步骤2的方法1
    def _find_image_in_md(self, md_content:str,image_file:str,context_len: int = 100) -> Tuple[str,str]:
        # 1、定义正则表达式
        # ![描述](images/文件名.扩展名)
        # r"字符串"：不要将其中的特殊符号进行转义
        # re.escape 转义图片文件名中的特殊字符，避免正则语法错误
        # .* 贪婪匹配 .*? 非贪婪匹配
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        # 2、找到1个匹配项即返回
        match = pattern.search(md_content)
        if not match:
            return None  # 没有找到
        start, end = match.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        return pre_text,post_text

    def _step_3_generate_summaries(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[str, str]:
        summaries = {}
        # 1、外部初始化双端队列，用于API速率限制，跨循环复用
        request_deque = deque()

        # 2、循环处理图片
        for img_file, image_path, context in target_images:
            # 2.1、速率限制
            self._apply_api_rate_limit(request_deque, max_requests=10)

            # 2.2、调用大模型生成图片摘要
            summaries[img_file] = self._summarize_image(image_path, root_folder=doc_stem, image_content=context)
            print(f"[摘要] {img_file} -> {summaries[img_file]!r}")

        return summaries
    # 步骤3方法一，时间戳限流
    def _apply_api_rate_limit(
            self,
            request_times: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ) -> None:
        """
        通用滑动窗口API速率限制器（抽离为公共工具）
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
        :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待
        """
        current_time = time.time()

        # 1. 清理滑动窗口外的过期请求时间戳，保证队列仅存窗口内的请求
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()

        # 2. 窗口内请求数达上限，计算并阻塞等待剩余时间
        if len(request_times) >= max_requests:
            # 计算需要等待的时长（窗口总时长 - 最早请求已存在的时长）
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info(
                    f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")
                time.sleep(sleep_duration)
                # 等待后更新当前时间，重新清理过期请求（避免等待期间有请求过期）
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()

        # 3. 记录当前请求时间戳，加入滑动窗口队列
        request_times.append(current_time)
        logging.getLogger().info(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")
    # 步骤三方法二，调用模型
    def _summarize_image(self, image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
        """
           调用多模态大模型总结图片内容。

           参数：
           - image_path: 图片本地路径。
           - root_folder: 文档所属文件夹名（提供更多上下文）。
           - image_content: 图片在文档中的上下文 (前文, 后文)。
        """
        # 图片的base64编码
        with open(image_path,"rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        # 1.vlm模型工具
        vl_ai = get_llm_client(lm_config.vl_model)

        # 2.调用模型
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{image_content[0]}"，下文部分为"{image_content[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。尽量不要超过10个字"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        response = vl_ai.invoke(messages)
        return response.content.strip().replace("\n","")

    def _step_4_upload_and_replace(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]],
                                   summaries: Dict[str, str], md_content: str) -> str:
        # 0.minio客户端
        minio_client = get_minio_client()
        # 总文件名
        minio_img_dir = minio_config.img_dir
        # 子文件名.stem——>提取文件名B530.pdf-->B530
        upload_dir = f"{minio_img_dir}/{doc_stem}"

        # 1.清理minio目录,保证没有重复的（非必要操作）
        self.clean_minio_directory(minio_client, upload_dir)

        # 2.批量上传图片，获得minio的urls
        urls = self.upload_images_batch(minio_client, upload_dir, target_images)


        # 3.将摘要和url路径合并
        image_info = self.merge_summary_and_url(summaries, urls)

        # 4.替换md文件的摘要和路径
        md_content = self.process_md_file(md_content, image_info)

        return md_content

    #步骤四方法一：清理目录，避免目录冲突
    def clean_minio_directory(self, minio_client: Minio, upload_dir) -> None:
        """
        幂等性清理：上传前先删除 MinIO 中指定目录下的旧文件。
        防止重名文件导致的内容混淆或垃圾堆积。
        :param minio_client: 初始化完成的MinIO客户端对象
        :param prefix: MinIO目录前缀（要清理的目录路径）
        """
        objects_to_delete = minio_client.list_objects(minio_config.bucket_name, prefix=upload_dir, recursive=True)
        # 构造删除列表
        delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
        if delete_list:
            errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            for error in errors:
                self.logger.error(f"删除失败：{error}")

    # 步骤四方法二：批量上传文件
    def upload_images_batch(self, minio_client: Minio, upload_dir: str, target_images: List[Tuple]) -> Dict[str, str]:
        """
        批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
        :param minio_client: 初始化完成的MinIO客户端对象
        :param upload_dir: MinIO上传根目录
        :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :return: 图片URL字典，键：图片文件名，值：MinIO访问URL
        """
        urls = {}
        # img_file：图片名
        for img_file,img_path,_ in target_images:
            # 上传
            object_name = f"{upload_dir}/{img_file}" #minio文件对象名（路径：带后缀）
            print(f"上传文件：{object_name}")
            urls[img_file] = self.upload_to_minio(minio_client, img_path, object_name)

        return urls


    # 步骤四方法三：合并参数
    def merge_summary_and_url(self, summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        """
        合并图片摘要字典和URL字典，过滤掉上传失败无URL的图片
        :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
        :param urls: 图片URL字典，键：图片文件名，值：MinIO访问URL
        :return: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)元组
        """
        image_info = {}
        for image_file,summary in summaries.items():
            image_info[image_file] = (summary, urls[image_file])
        return image_info

    # 步骤四方法五：替换md中url和摘要summary
    def process_md_file(self, md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
        """
        核心功能：替换MD内容中的本地图片引用为MinIO远程引用
        替换规则：![原描述](本地路径) → ![图片摘要](MinIO访问URL)
        :param md_content: 原始MD文件内容
        :param image_info: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)
        :return: 替换后的新MD内容
        """
        for image_file,(summary, url) in image_info.items():
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
            md_content = pattern.sub(lambda m: f"![{summary}]({url})", md_content)
        return md_content


    # 步骤四方法2：上传minio
    def upload_to_minio(self, minio_client: Minio, img_path: str, object_name: str) -> str | None:
        """
        将单张本地图片上传至MinIO对象存储，并返回公网可访问URL
        :param minio_client: 初始化完成的MinIO客户端对象
        :param local_path: 图片本地完整路径
        :param object_name: MinIO中要存储的对象名称
        :return: 图片MinIO访问URL（上传失败返回None）
        """
        ifSuccess = minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=img_path,
            content_type=f"image/{os.path.splitext(img_path)[1][1:]}"
        )
        url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"
        return url
    # 步骤五：保存备份后的新文档
    def _step_5_backup_new_md_file(self, origin_md_path: str, md_content: str) -> str:
        """
        步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
        新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
        :param origin_md_path: 原始MD文件完整路径
        :param md_content: 处理后的新MD内容
        :return: 新MD文件的完整路径
        """
        # 构造新文件路径：替换原后缀为 _new.md
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"

        # 写入新MD内容（覆盖写入，若文件已存在则更新）
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.logger.info(f"处理后MD文件已保存，新文件路径：{new_md_file_name}")

        return new_md_file_name


if __name__ == '__main__':
    setup_logging()
    init_state = {
        "md_path":r"E:\AI Agent\g0525\output\华为擎云B530 用户指南-(PUCZ,Windows11_03,zh-cn)\华为擎云B530 用户指南-(PUCZ,Windows11_03,zh-cn).md",
        "md_content": None
    }
    node = NodeMDImg()
    result_state = node(init_state)
    dumps = json.dumps(result_state, ensure_ascii=False, indent=2)
    print(dumps)