本项目是关于Yi的多模态系列模型，如Yi-VL-6B/34B等的实验与应用。

### 模型推理

以命令行（CLI）的模型进行模型推理，需要将图片下载至images文件夹，同时将`single_inference.py`略作调整，以支持多次提问。

运行命令如下：

```bash
CUDA_VISIBLE_DEVICES=0 python single_inference.py --model-path /data-ai/usr/models/Yi-VL-34B --image-file images/cats.jpg --question "How many cats are there in this image?"
```

模型推理时使用一张A100（显存80G）就可满足推理要求。

示例图片如下：

![示例图片](https://s2.loli.net/2024/01/24/ZvF5dTAL8b3oSah.jpg)

回复结果如下：

![Yi-VL-34B模型回复](https://s2.loli.net/2024/01/24/6aqLIsl1Ted92Kn.png)


### 可视化模型问答

基于此，我们将会用gradio模块，对`Yi-VL-34B`模型和`GPT-4V`模型的结果进行对比。

Python代码参考`gradio_server.py`.

以下是对不同模型和问题的回复：

- 图片：taishan.jpg，问题：这张图片是中国的哪座山？

![](https://s2.loli.net/2024/01/25/R5lDfZrW6BVkdzN.png)

- 图片：dishini.jpg，问题：这张图片是哪个景点的logo？

![](https://s2.loli.net/2024/01/25/sjpBTodKwfnvuJ4.png)

- 图片：fruit.jpg，问题：详细描述下这张图片

![](https://s2.loli.net/2024/01/25/VsFvE32PrmZQ6Yd.png)

- 图片：football.jpg，问题：图片中一个有几个人，他们在干什么？

![](https://s2.loli.net/2024/01/25/GthKWBfiIjmd8wT.png)

- 图片：cartoon.jpg，问题：这张图片是哪部日本的动漫？

![](https://s2.loli.net/2024/01/25/9o1jMaQTK3c8bqH.png)

从以上的几个测试用例来看，`Yi-VL-34B`模型的效果很不错，但对比`GPT-4V`模型，不管在图片理解，还是模型的回答上，仍有一定的差距。

最后，我们来看一个验证码的例子（因为GPT-4V是不能用来破解验证码的！）

![](https://s2.loli.net/2024/01/25/eZimVOFIEAcL7hy.png)

可以看到，`Yi-VL-34B`模型在尝试回答，但给出了错误答案，而`GPT-4V`模型则会报错，报错信息如下：

```json
{
  "error": {
    "message": "Your input image may contain content that is not allowed by our safety system.",
    "type": "invalid_request_error",
    "param": null,
    "code": "content_policy_violation"
  }
}
```

无疑，`GPT-4V`模型这样的设计是合情合理的。