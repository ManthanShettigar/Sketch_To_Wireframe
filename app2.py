import gradio as gr
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from generator_model import Generator
import config

# Load the generator model
gen = Generator(in_channels=3, features=64).to(config.DEVICE)
checkpoint = torch.load(config.CHECKPOINT_GEN, map_location=config.DEVICE)
gen.load_state_dict(checkpoint['state_dict'])


# Define the prediction function
def generate_fake_image(input_image):
    # Convert the input image to a PyTorch tensor
    
    input_tensor = to_tensor(input_image).unsqueeze(0)
    input_tensor=input_tensor.to(config.DEVICE)
    print(input_tensor.shape)
    # Generate the fake image using the generator model
    with torch.autograd.detect_anomaly():
        output_tensor = gen(input_tensor).squeeze()
        print(output_tensor.shape)
    # Convert the output tensor to a PIL image
    output_image = to_pil_image(output_tensor)

    # Return the output image as a NumPy array
    return output_image

# Define the Gradio interface
inputs = gr.inputs.Image()
outputs = gr.outputs.Image(type='pil')
app = gr.Interface(fn=generate_fake_image, inputs=inputs, outputs=outputs)

# Launch the app
app.launch()
