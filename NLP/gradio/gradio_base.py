import gradio as gr

def greet(name):
    return "Hello " + name + "!"

class GradioBase:
    def __init__(self, main_func, inputs_type, outputs_type):
        self.inputs_type = inputs_type
        self.outputs_type = outputs_type

        if self.inputs_type == "text":
            self.inputs = gr.inputs.Textbox(lines=2, label="Input Text")
        if self.outputs_type == "text":
            self.outputs = gr.outputs.Textbox(label="Output Text")

        self.main_func = main_func
        self.demo = gr.Interface(fn=self.main_func, inputs=self.inputs, outputs=self.outputs)

    def launch(self):
        self.demo.launch()

    