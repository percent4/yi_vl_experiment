import os
import traceback

import torch
import requests
import json
import base64
import gradio as gr
from PIL import Image
from datetime import datetime
import logging

from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    load_pretrained_model,
    get_model_name_from_path,
    tokenizer_image_token
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

disable_torch_init()
model_path = "/data-ai/usr/lmj/models/Yi-VL-34B"
key_info["model_path"] = model_path
get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path)
print("model loaded!")


def model_infer(qs, image_file, temperature, top_p, max_output_tokens):
    global model
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates["mm_default"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    image = Image.open(image_file)
    if getattr(model.config, "image_aspect_ratio", None) == "pad":
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean)
        )
    image_tensor = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ][0]

    stop_str = conv.sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    model = model.to(dtype=torch.bfloat16)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
            stopping_criteria=[stopping_criteria],
            max_new_tokens=max_output_tokens,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


def gpt_4v_answer(question, image_path, temperature, top_p, max_output_tokens):
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        api_key = os.environ["OPENAI_API_KEY"]
        url = "https://api.openai.com/v1/chat/completions"

        payload = json.dumps({
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p
        })
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        result = response.json()["choices"][0]["message"]["content"]
    except Exception as err:
        print(traceback.format_exc())
        print(response.text)
        result = "something error with gpt-4v"

    return result


def model_answer(model_list, question, array, temperature, top_p, max_output_tokens):
    # save image
    im = Image.fromarray(array)
    image_path = f"./images/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg"
    im.save(image_path)
    model_output, gpt4v_output = "", ""
    if "Yi-VL-34B" in model_list:
        # yi-vl-34b output
        model_output = model_infer(question, image_path, temperature, top_p, max_output_tokens)
    if "GPT-4V" in model_list:
        # gpt-4v output
        gpt4v_output = gpt_4v_answer(question, image_path, temperature, top_p, max_output_tokens)
    logger.info(f"image path: {image_path}, question: {question}\n"
                f"yi-vl-34 output: {model_output}\n"
                f"gpt-4v output: {gpt4v_output}\n")
    return model_output, gpt4v_output


if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                checkout_group = gr.CheckboxGroup(choices=["Yi-VL-34B", "GPT-4V"], value="Yi-VL-34B", label='models')
                image_box = gr.inputs.Image()
                user_input = gr.TextArea(lines=5, placeholder="your question about the image")
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                        label="Temperature", )
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P", )
                max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True,
                                              label="Max output tokens", )
            with gr.Column():
                yi_vl_output = gr.TextArea(lines=5, label='Yi-VL-34B')
                gpt_4v_output = gr.TextArea(lines=5, label='GPT-4V')
                submit = gr.Button("Submit")
        submit.click(fn=model_answer,
                     inputs=[checkout_group, user_input, image_box, temperature, top_p, max_output_tokens],
                     outputs=[yi_vl_output, gpt_4v_output])

    demo.launch(server_port=50072, server_name="0.0.0.0", share=True)
