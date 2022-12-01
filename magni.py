#!/usr/bin/env python
"""Magni."""
# HTTPS_PROXY=socks5h://127.0.0.1:9995 ./magni.py --url https://chapmanganato.com/manga-dt980702/chapter-184

import argparse
import asyncio
import concurrent.futures
import http.server
import os
import random
import socketserver
import sys
import typing

import bs4
import cv2  # type:ignore
import jinja2
import requests


class Argparser:  # pylint: disable=too-few-public-methods
    """Argparser class."""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--url",
            "-u",
            type=str,
            help="the url to the page containing the images",
            default=False,
        )
        self.parser.add_argument(
            "--method",
            "-m",
            type=str,
            help="the method to use. either fsrcnn or espcn",
            default="espcn",
        )
        self.parser.add_argument(
            "--port",
            "-p",
            type=int,
            help="the port to serve the images over",
            default=8086,
        )
        self.args = self.parser.parse_args()


def get_manganato_headers(url: str) -> typing.Dict[str, str]:
    """Sets the ncessary headers."""
    headers = {
        "Accept": "image/avif,image/webp,*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
        "Sec-GPC": "1",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "TE": "trailers",
        "Referer": url,
        "User-Agent": get_user_agent(),
    }

    return headers


def get_model_path() -> str:
    """Get the model path."""
    model_path: str = ""
    if "MAGNI_MODEL_PATH" in os.environ and os.environ["MAGNI_MODEL_PATH"]:
        model_path = os.environ["MAGNI_MODEL_PATH"]
    else:
        model_path = "./models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


def get_image_path() -> str:
    """Get the image path."""
    image_path: str = ""
    if "MAGNI_IMAGE_PATH" in os.environ and os.environ["MAGNI_IMAGE_PATH"]:
        image_path = os.environ["MAGNI_IMAGE_PATH"]
    else:
        image_path = "./images"

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    return image_path


# TODO-both models are garbage. should train models specifically
# for black and white pics.
def espcn_superscaler(img):
    """ESPCN superscaler."""
    superres = cv2.dnn_superres.DnnSuperResImpl_create()
    path = get_model_path() + "/" + "ESPCN_x3.pb"
    superres.readModel(path)
    superres.setModel("espcn", 3)
    result = superres.upsample(img)
    return result


def fsrcnn_superscaler(img):
    """FSRCNN superscaler"""
    superres = cv2.dnn_superres.DnnSuperResImpl_create()
    path = get_model_path() + "/" + "FSRCNN_x3.pb"
    superres.readModel(path)
    superres.setModel("fsrcnn", 3)
    result = superres.upsample(img)
    return result


# flake8: noqa: E501
def get_user_agent() -> str:
    """Returns a random user agent."""
    # user_agents = [
    #     "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
    #     "Mozilla/5.0 (Windows NT 10.0; rv:78.0) Gecko/20100101 Firefox/78.0",
    #     "Mozilla/5.0 (X11; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0",
    #     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    # ]
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    ]

    return user_agents[random.randint(0, len(user_agents) - 1)]


def get_proxies() -> typing.Dict:
    """Get the proxy env vars."""
    http_proxy: typing.Optional[str] = None
    if "HTTP_PROXY" in os.environ and os.environ["HTTP_PROXY"] != "":
        http_proxy = os.environ["HTTP_PROXY"]

    https_proxy: typing.Optional[str] = None
    if "HTTPS_PROXY" in os.environ and os.environ["HTTPS_PROXY"] != "":
        https_proxy = os.environ["HTTPS_PROXY"]

    no_proxy: typing.Optional[str] = None
    if "NO_PROXY" in os.environ and os.environ["NO_PROXY"] != "":
        no_proxy = os.environ["NO_PROXY"]

    return {"http": http_proxy, "https": https_proxy, "no_proxy": no_proxy}


def single_get(url: str) -> requests.Response:
    """A simple get."""
    return requests.get(
        url,
        allow_redirects=True,
        timeout=10,
        proxies=get_proxies(),
        headers=get_manganato_headers(url),
    )


def multi_get(urls: list) -> list:
    """Async get."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        response_list = list(pool.map(single_get, urls))
    return response_list


def single_get_tag(url_tag_pair: list) -> typing.Tuple[requests.Response, str]:
    """A simple get with a tag."""
    return (
        requests.get(
            url_tag_pair[0],
            allow_redirects=True,
            timeout=10,
            proxies=get_proxies(),
            headers=get_manganato_headers(url_tag_pair[0]),
        ),
        url_tag_pair[1],
    )


def multi_get_tag(
    urls: typing.List[typing.Tuple[str, str]],
) -> typing.Optional[typing.List[typing.Tuple[requests.Response, str]]]:
    """Async get with a tag."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        response_list = list(pool.map(single_get_tag, urls))
    return response_list


# flake8: noqa: E501
async def model_downloader() -> typing.Optional[
    typing.List[typing.Tuple[str, str]]
]:
    """Download the models."""
    down_list = [
        "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb",
        "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb",
        "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb",
        "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
        "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb",
        "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
    ]

    url_list: typing.List[str] = []
    for model in down_list:
        if (
            os.path.exists(
                get_model_path() + "/" + model[model.rfind("/") + 1 :]
            )
            is False
        ):
            url_list.append(model)

    url_tag_list: typing.List[typing.Tuple[str, str]] = []
    for url in url_list:
        url_tag_list.append((url, url[url.rfind("/") + 1 :]))
    response_list: typing.Optional[
        typing.List[typing.Tuple[requests.Response, str]]
    ] = multi_get_tag(url_tag_list)

    model_path: str = os.getcwd()
    if (
        "MAGNI_MODEL_PATH" in os.environ
        and os.environ["MAGNI_MODEL_PATH"] != ""
    ):
        model_path = os.environ["MAGNI_MODEL_PATH"]

    if response_list is None:
        return None
    for response, name in response_list:
        with open(model_path + "/" + name, mode="b+w") as downed:
            downed.write(response.content)

    return url_tag_list


async def download_all_images(url: str) -> typing.Optional[typing.List[str]]:
    """Sniffs images."""
    response = requests.get(url, timeout=10, allow_redirects=True)
    if response.content is None:
        return None

    soup = bs4.BeautifulSoup(response.content, "lxml")
    search_results = soup.findAll("img")

    image_url_list: typing.List[typing.Tuple[str, str]] = [
        (result["src"], result["src"][result["src"].rfind("/") + 1 :])
        for result in search_results
    ]

    print(image_url_list)
    response_list: typing.Optional[
        typing.List[typing.Tuple[requests.Response, str]]
    ] = multi_get_tag(image_url_list)
    print(response_list)

    if response_list is None:
        return None

    image_name_list: typing.List[str] = []
    for response, name in response_list:
        image_name_list.append(name)
        with open(get_image_path() + "/" + name, "w+b") as image:
            image.write(response.content)

    return image_name_list


def superres_images(image_list: typing.List[str], method: str) -> None:
    """Superscales the images."""
    for image in image_list:
        img = cv2.imread(get_image_path() + "/" + image)
        if method == "espcn":
            result = espcn_superscaler(img)
        elif method == "fsrcnn":
            result = fsrcnn_superscaler(img)
        cv2.imwrite(get_image_path() + "/" + image, result)


async def handle_downloads(
    argparser: Argparser,
) -> typing.Optional[typing.List[str]]:
    """Download the models and the images."""
    _, image_name_list = await asyncio.gather(
        model_downloader(), download_all_images(argparser.args.url)
    )

    return image_name_list


def fill_jinja_template(image_name_list: typing.List[str]) -> None:
    """Fills the jinja template."""
    environment = jinja2.Environment(
        autoescape=True,
        loader=jinja2.FileSystemLoader(os.getcwd()),
    )
    template = environment.get_template(os.path.join("template.jinja2"))
    temp_head = template.render({"image_list": image_name_list})
    with open(
        get_image_path() + "/" + "index.html", encoding="utf-8", mode="w"
    ) as out_file:
        out_file.write(temp_head)


class MagniHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """We want to server our own index.html from an arbitrary location."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=get_image_path(), **kwargs)


# TODO-add graceful shutdown
def serve(port: int) -> None:
    """Startup a simple http file server."""
    handler = MagniHTTPRequestHandler

    print(f"now servering on 127.0.0.1:{repr(port)}")
    with socketserver.TCPServer(("", port), handler) as httpd:
        httpd.serve_forever()


def main() -> None:
    """Entry point."""
    argparser = Argparser()
    image_name_list = asyncio.run(handle_downloads(argparser))

    if image_name_list is not None:
        superres_images(image_name_list, argparser.args.method)
        print("finished superresing images.")
    else:
        print("failed to download all images.", file=sys.stderr)
        sys.exit(1)
    fill_jinja_template(image_name_list)
    serve(argparser.args.port)


if __name__ == "__main__":
    main()
