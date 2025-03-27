# Real Time Translate by sensvoice or vosk ASR and stranslate function (local version)

## 備忘錄
現在主力推 funASR 的 sensvoice + Google 的 Gemma 
目前的 Docker 是可以做到這件事，但他自己載的 CT2 會有問題感覺要從原始碼重包
那 CT2 有問題代表 argos 就不能用，所以以現在這板包的 argos 會不能用
把 docker 換成原本註解掉的那個 arogs 可以用蛋 gemma 就不能因為 pytohn 3.8 的 transformers 只支援到 4.46.3 而 Gemma 要求要 4.50.1 (python > 3.9)
AGX 我們 jetpack 版本 5.1.2 默認支援的都是 python 3.8，官方最新的只到3.8沒更新的了，現在包的版本會可以用是因為不是官方的他的 python 是 3.10 且 pytorhc 吃的到 CUDA
這個問題之後要解決

## Now support language

```
["ZH", "EN", "JA", "KO"]
```

## This is the local version
