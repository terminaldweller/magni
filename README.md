# magni
magni takes a url and upscales all the images inside and returns a simple html page with the images embedded.</br>

```sh
HTTPS_PROXY=socks5h://127.0.0.1:9094 ./magni.py --url https://chapmanganato.com/manga-dt980702/chapter-184
```

## Notes
* `magni` can use a socks5 proxy as is displayed in the above example.</br>
* currently the models we are using are not as effective. I should either fine ones that are specifically trained on greyscale images or just train some myself.</br>
