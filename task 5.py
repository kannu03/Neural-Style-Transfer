import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration
print("step 1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load an image
print("step 2")
def load_image(img_path, transform=None, max_size=400, shape=None):
    image = Image.open(img_path)

    # Resize the image if it's larger than max_size
    print("step 3")
    if max_size is not None:
        size = max(max_size, image.size[0])
        if shape is not None:
            size = shape
        image = image.resize((size, int(size * image.size[1] / image.size[0])), Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image.to(device)

# Image preprocessing
print("step 4")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Load content and style images with correct file paths
print("step 5")
content_image = load_image("/content/sample_data/watch 1.jpg", transform=transform).to(device)
style_image = load_image("/content/sample_data/watch 1.jpg", transform=transform, shape=content_image.shape[-2:][0]).to(device)

# Display images function
print("step 6")
def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)

    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

# Load the VGG19 model
print("step 7")
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Define the layers to use for content and style features
print("step 8")
content_layers = {21: 'conv_4'}  # conv_4 corresponds to layer 21 in VGG19
style_layers = {0: 'conv_1', 5: 'conv_2', 10: 'conv_3', 19: 'conv_4', 28: 'conv_5'}  # these are the style layers

# Extract features function
print("step 9")
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if int(name) in layers:
            features[layers[int(name)]] = x
    return features

# Gram Matrix function to calculate style representation
print("step 10")
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Get content and style features
content_features = get_features(content_image, vgg, layers=content_layers)
style_features = get_features(style_image, vgg, layers=style_layers)

# Create a target image and clone the content image
print("step 11")
target = content_image.clone().requires_grad_(True).to(device)

# Define the optimizer
optimizer = optim.Adam([target], lr=0.003)

# Define style and content weights
style_weight = 1e6
content_weight = 1

# Perform style transfer
print("step 12")
for i in range(1, 301):  # Reduced iterations for quicker results
    target_features = get_features(target, vgg, layers={**content_layers, **style_layers})

    # Content loss
    content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4'])**2)

    # Style loss
    print("step 13")
    style_loss = 0
    for layer in style_layers.values():
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])
        layer_style_loss = torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / len(style_layers)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    optimizer.step()

    if i % 50 == 0:  # Display output every 50 iterations
        print(f"Iteration {i}, Total Loss: {total_loss.item()}")
        imshow(target, title=f'Output Image at Iteration {i}')

# Display the final output
print("step 14")
imshow(target, title='Final Output Image')
