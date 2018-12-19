# -*- coding: utf-8 -*-

import logging

def get_logger(path=None):

    # 格式设置
    fm = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")  # 设置输出日志格式
    fm.datefmt = "%Y-%m-%d %H:%M:%S" # 设置时间格式

    logger = logging.getLogger()  # 获得一个logger对象，默认是root

    # 输出日志到控制台
    ch = logging.StreamHandler()
    ch.setFormatter(fm)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)  # 把文件流添加进来，流向写入到控制台

    # 输出日志到文件
    if not path==None:
        fh = logging.FileHandler(path, encoding='utf-8')  # 创建一个文件流并设置编码utf8
        fh.setFormatter(fm)  # 把文件流添加写入格式
        fh.setLevel(logging.INFO) # 设置最低等级debug
        logger.addHandler(fh) # 把文件流添加进来，流向写入到文件

    logger.setLevel(logging.DEBUG)

    return logger

