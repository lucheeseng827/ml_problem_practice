import torch
import torchvision
from torchvision import models
from torch.quantization import get_default_qconfig, prepare, convert

# Load your model
model = torch.load("path_to_your_model.pth")
model.eval()

# Set the backend for quantized operations
torch.backends.quantized.engine = "qnnpack"

# Specify the quantization configuration
qconfig = get_default_qconfig("qnnpack")

# Prepare the model for static quantization. This inserts observers in the model that will compute
# scaling parameters to be used in the subsequent quantization step.
model_prepared = prepare(model, inplace=False)
model_prepared.qconfig = qconfig

# Calibrate the model with representative data
# NOTE: You'll need to run the model with some representative data samples
# for calibration_data in your_data_loader:
#     model_prepared(calibration_data)

# Convert the model to a quantized version
model_quantized = convert(model_prepared, inplace=False)

# Save the quantized model
torch.jit.save(torch.jit.script(model_quantized), "quantized_model.pth")

print("Quantized model saved!")
