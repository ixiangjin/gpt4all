from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="LLukas22/gpt4all-lora-quantized-ggjt", filename="ggjt-model.bin", local_dir=".")
llm = Llama(model_path="./ggjt-model.bin")


ins = '''### Instruction:
{}
### Response:
'''

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)




# def generate(instruction): 
#     response = llm(ins.format(instruction))
#     response = response['choices'][0]['text']
#     result = ""
#     for word in response.split(" "):
#         result += word + " "
#         yield result

def generate(instruction): 
    result = ""
    for x in llm(ins.format(instruction), stop=['### Instruction:', '### End'], stream=True):
        result += x['choices'][0]['text']
        yield result


examples = [
    # "Instead of making a peanut butter and jelly sandwich, what else could I combine peanut butter with in a sandwich? Give five ideas",
    # "How do I make a campfire?",
    # "Explain to me the difference between nuclear fission and fusion.",
    # "I'm selling my Nikon D-750, write a short blurb for my ad."
]

def process_example(args):
    for x in generate(args):
        pass
    return x
    
css = ".generating {visibility: hidden}"

# Based on the gradio theming guide and borrowed from https://huggingface.co/spaces/shivi/dolly-v2-demo
class SeafoamCustom(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            input_background_fill="zinc",
            input_border_color="*secondary_300",
            input_shadow="*shadow_drop",
            input_shadow_focus="*shadow_drop_lg",
        )


seafoam = SeafoamCustom()


with gr.Blocks(theme=seafoam, analytics_enabled=False, css=css) as demo:
    with gr.Column():
        gr.Markdown(
            """ ## GPT 2 gen sql
            
            Type in the box below and click the button to generate answers to your most pressing questions!
            
      """
        )

        with gr.Row():
            with gr.Column(scale=3):
                instruction = gr.Textbox(placeholder="Enter your question here", label="Question", elem_id="q-input")

                with gr.Box():
                    gr.Markdown("**Answer**")
                    output = gr.Markdown(elem_id="q-output")
                submit = gr.Button("Generate", variant="primary")
                gr.Examples(
                    examples=examples,
                    inputs=[instruction],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )
        


    submit.click(generate, inputs=[instruction], outputs=[output])
    instruction.submit(generate, inputs=[instruction], outputs=[output])

demo.queue(concurrency_count=1).launch(debug=True)