# magni
magni takes a url and upscales all the images inside and returns a simple html page with the images embedded.</br>

```sh
usage: magni.py [-h] [--url URL] [--method METHOD] [--port PORT]

options:
  -h, --help            show this help message and exit
  --url URL, -u URL     the url to the page containing the images
  --method METHOD, -m METHOD
                        the method to use. either fsrcnn or espcn
  --port PORT, -p PORT  the port to serve the images over
```

## Install and Run

### Install
```sh
poetry install
```

### Run
```sh
poetry shell && HTTPS_PROXY=socks5h://127.0.0.1:9094 ./magni.py --url https://chapmanganato.com/manga-dt980702/chapter-184
```
you can obviously use `poetry run` as well:
```sh
HTTPS_PROXY=socks5h://127.0.0.1:9094 poetry run ./magni.py --url https://chapmanganato.com/manga-dt980702/chapter-184
```

## Env Vars
magni recognizes three environment variables:</br>

### HTTPS_PROXY
You must specify a socks5 proxy here since magni uses `pysocks` to make the connections.</br>
If the env var is not defined magni will not use any proxy.</br>

### MAGNI_MODEL_PATH
Path to the directory where magni will store the model.</br>
If the env var is not defined, magni will use `./models` as a default value.</br>

### MAGNI_IMAGE_PATH
Path to the directory where magni will store the upscaled images.</br>
If the env var is not defined or empty magni will use `./images` as a default value.</br>

## TODO
* currently the models we are using are not as effective. I should either fine ones that are specifically trained on greyscale images or just train some myself.</br>
