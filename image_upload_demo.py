import gradio as gr


# 定义处理图片上传的函数
def process_image(uploaded_image):
    # 这里可以添加处理图片的代码
    # 现在我们只是简单地返回上传的图片
    print(uploaded_image)
    return "hello"


# 创建Gradio界面，定义输入和输出
iface = gr.Interface(
    fn=process_image,  # 处理上传图片的函数
    inputs=gr.inputs.Image(),  # 定义输入为图片，可以设置图片的shape
    outputs="text",  # 定义输出也为图片
    title="图片上传页面",  # 页面标题
    description="上传图片，并查看结果。"  # 页面描述
)

# 运行Gradio应用
iface.launch()
