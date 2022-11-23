#!/usr/bin/env python
"""Magni."""

import argparse
import concurrent.futures
import http.server
import os
import socketserver
import typing

import bs4
import cv2  # type:ignore
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
            "--port",
            "-p",
            type=int,
            help="the port to serve the images over",
            default=6666,
        )
        self.args = self.parser.parse_args()


def get_model_path() -> str:
    """Get the model path."""
    # FIXME- add path if it doesnt exist
    if "MAGNI_MODEL_PATH" in os.environ and os.environ["MAGNI_MODEL_PATH"]:
        return os.environ["MAGNI_MODEL_PATH"]
    return "./models"


def get_image_path() -> str:
    """Get the image path."""
    # FIXME- add path if it doesnt exist
    if "MAGNI_IMAGE_PATH" in os.environ and os.environ["MAGNI_IMAGE_PATH"]:
        return os.environ["MAGNI_IMAGE_PATH"]
    return "./images"


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
        url, allow_redirects=True, timeout=10, proxies=get_proxies()
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


def model_downloader() -> None:
    """Download the models."""
    down_list = [
        "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb",
        "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb",
        "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb",
        "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
        "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb",
        "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
    ]

    url_tag_list: typing.List[typing.Tuple[str, str]] = []
    for url in down_list:
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

    return None


def download_all_images(url: str) -> None:
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

    for response, name in response_list:
        with open(get_image_path() + "/" + name, "w+b") as image:
            image.write(response.content)

    return None


def serve(port_number: int) -> None:
    """Startup a simple http file server."""
    handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", port_number), handler) as httpd:
        httpd.serve_forever()


def main() -> None:
    """Entry point."""
    argparser = Argparser()
    model_downloader()
    download_all_images(argparser.args.url)


if __name__ == "__main__":
    main()
