import gradio as gr

def echo(x):
    return x

demo = gr.Interface(fn=echo, inputs="text", outputs="text")
demo.launch(server_name="127.0.0.1", server_port=7861, share=False)