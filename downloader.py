import argparse
import os
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock

from requests import get, head


lock = Lock()


class Downloader:
    def __init__(self, url: str, thread_nums: int, output_file: str):
        self.url = url
        self.thread_nums = thread_nums
        self.output_file = output_file
        r = head(self.url)
        # 若资源显示302,则迭代找寻源文件
        while r.status_code == 302:
            self.url = r.headers["Location"]
            print(f"该url已重定向至{self.url}")
            r = head(self.url)
        self.size = int(r.headers["Content-Length"])

    def down(self, start, end):
        headers = {"Range": f"bytes={start}-{end}"}
        # stream = True 下载的数据不会保存在内存中
        r = get(self.url, headers=headers, stream=True)
        # 写入文件对应位置,加入文件锁
        lock.acquire()
        with open(self.output_file, "rb+") as fp:
            fp.seek(start)
            fp.write(r.content)
            lock.release()
            # 释放锁

    def run(self):
        # 创建一个和要下载文件一样大小的文件
        fp = open(self.output_file, "wb")
        fp.truncate(self.size)
        fp.close()
        # 启动多线程写文件
        print(f"The file has {self.size / 1024 / 1024:.2f} MB.")

        part = self.size // self.thread_nums
        pool = ThreadPoolExecutor(max_workers=self.thread_nums)
        futures = []
        for i in range(self.thread_nums):
            start = part * i
            # 最后一块
            if i == self.thread_nums - 1:
                end = self.size
            else:
                end = start + part - 1
                print(f"Thread {i} start:{start} end:{end}")
            futures.append(pool.submit(self.down, start, end))
        wait(futures)
        print(f"Download finished! Save to {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url-file",
        type=str,
        default="/data/Users/xyq/developer/happy_code/data/vpt/10xx/20240726_10xx_actions_url.txt",
    )
    parser.add_argument("--thread-nums", type=int, default=16)
    parser.add_argument("--save-dir", type=str, default="data/vpt/10xx/action")

    args = parser.parse_args()

    with open(args.url_file) as f:
        urls = f.readlines()
    # print(urls)
    error_download_file = open("vpt_download_error.txt", "a")
    for url in urls:
        url = url.strip()
        output_path = os.path.join(args.save_dir, url.split("/")[-1])
        if os.path.exists(output_path):
            continue
        try:
            downloader = Downloader(url, args.thread_nums, output_path)
            downloader.run()
        except Exception as e:
            print(f"Download {url} error: {e}")
            error_download_file.write(url)
            error_download_file.write("\n")
            error_download_file.flush()
        # break

    print("All download finished!")
