#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gc
import os

# Make tensorflow not take over the entire GPU memory

if torch.cuda.is_available():
    # Set memory growth behavior (manually or automatically managed by CUDA)
    for i in range(torch.cuda.device_count()):
        # Set the memory fraction (optional, defaults to 1.0 meaning use all available memory)
        torch.cuda.set_per_process_memory_fraction(1.0, i)
        
        # Optionally, you can also clear unused memory (this is the closest thing to memory growth in PyTorch)
        torch.cuda.empty_cache()

from torchga.torchga import GeometricAlgebra
from torchga.layers import GeometricProductDense, GeometricSandwichProductDense, TensorToGeometric, GeometricToTensor
from torchga.layers import GeometricProductElementwise, GeometricSandwichProductElementwise


# In[2]:


def make_batch(batch_size):
    triangle_points = 2 * torch.rand((batch_size, 3, 2)) - 1
    x, y = triangle_points[..., 0], triangle_points[..., 1]
    ax, ay, bx, by, cx, cy = x[..., 0], y[..., 0], x[..., 1], y[..., 1], x[..., 2], y[..., 2]
    triangle_areas = 0.5 * torch.abs(ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    return triangle_points, triangle_areas

num_samples = 3
sample_points, sample_areas = make_batch(num_samples)

fig, axes = plt.subplots(1, num_samples, figsize=(12, 4), sharex=True, sharey=True)
for i, ax in enumerate(axes):
    points = sample_points[i]
    area = sample_areas[i]
    center = torch.mean(points, dim=0)
    ax.scatter(points[..., 0], points[..., 1])
    ax.add_patch(plt.Polygon(points))
    ax.annotate("Area: %.2f" % area, center)
fig.show()


# In[3]:
gc.collect()
torch.cuda.empty_cache() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}', flush = True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


ga = GeometricAlgebra([1, 1])
s_indices = [0]
v_indices = [1, 2]
mv_indices = torch.arange(0, ga.num_blades)

s_indices = torch.tensor(s_indices).to(device)
v_indices = torch.tensor(v_indices).to(device)
mv_indices = mv_indices.to(device)

# Define the PyTorch model, analogous to tf.keras.Sequential
model1 = TensorToGeometric(ga, blade_indices=v_indices)
model2 = GeometricProductDense(
        ga, num_input_units=3, num_output_units=64, activation="relu",
        blade_indices_kernel=mv_indices,
        blade_indices_bias=mv_indices
    )



# Example input
sample_points = torch.randn([num_samples, 3, 2]).to(device)

# In[6]:


model = nn.Sequential(
    TensorToGeometric(ga, blade_indices=v_indices),
    GeometricProductDense(
        ga, num_input_units=3, num_output_units=64, activation="relu",
        blade_indices_kernel=mv_indices,
        blade_indices_bias=mv_indices
    ),
    GeometricProductDense(
        ga, num_input_units=64, num_output_units=64, activation="relu",
        blade_indices_kernel=mv_indices,
        blade_indices_bias=mv_indices
    ),
    GeometricProductElementwise(
        ga, num_input_units=64, num_output_units=64, activation="relu",
        blade_indices_kernel=mv_indices,
        blade_indices_bias=mv_indices
    ),
    GeometricProductDense(
        ga, num_input_units=64, num_output_units=1,
        blade_indices_kernel=mv_indices,
        blade_indices_bias=s_indices
    ),
    GeometricToTensor(ga, blade_indices=s_indices)
)

model = model.to(device)

# Print samples and model output
#
print("Samples:", sample_points[0])
#print("Model(Samples):", model[0](sample_points).shape)
print("Model(Samples):", model(sample_points).shape)


# ## Train Loop, Fix Sandwich Product Layers

# In[8]:



from torch.utils.data import DataLoader, TensorDataset

train_points, train_areas = make_batch(1024)
test_points, test_areas = make_batch(128)



# Create datasets for training and validation
train_dataset = TensorDataset(train_points, train_areas)
test_dataset = TensorDataset(test_points, test_areas)

# Create DataLoaders with batch_size=32
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# In[9]:


import torch.nn as nn
import torch.optim as optim

# Assuming MyModel is already defined
optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()  # Assuming this is a regression task
epochs = 200


# In[10]:


train_points.shape


# In[ ]:



for epoch in range(epochs):
    model.train()
    running_loss = 0.0  # Track training loss for the epoch

    # Training loop over batches
    for batch_points, batch_areas in train_loader:
        #print(batch_points[0])
        #print(batch_areas[0])
        #print("***")
        optimizer.zero_grad()

        batch_points = batch_points.to(device)
        batch_areas = batch_areas.to(device)
        
        # Forward pass
        outputs = model(batch_points)
        outputs = outputs.squeeze()
        #print(outputs.shape)
        loss = criterion(outputs, batch_areas)
        
        # Backward pass and optimization
        loss.backward()
        '''
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"No gradient for {name}")
            else:
                print(f"{name} gradient mean: {param.grad.mean()}")
                print(f"{name} gradient max: {param.grad.mean()}")
                print(" ")
        '''
        optimizer.step()
        
        running_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_points, val_areas in test_loader:
            val_outputs = model(val_points.to(device))
            val_outputs = val_outputs.squeeze()
            batch_val_loss = criterion(val_outputs, val_areas.to(device))
            val_loss += batch_val_loss.item()
    
    # Calculate average loss per batch for training and validation
    running_loss /= len(train_loader)
    val_loss /= len(test_loader)
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}")


# In[ ]:


num_samples = 10
sample_points, sample_areas = make_batch(num_samples)
sample_points = sample_points.to(device)

predicted_sample_areas = model(sample_points)
#print(predicted_sample_areas.shape)
predicted_sample_areas = predicted_sample_areas.squeeze()
#print(predicted_sample_areas.shape)

fig, axes = plt.subplots(1, num_samples, figsize=(20, 5), sharex=True, sharey=True)
for i, ax in enumerate(axes):
    points = sample_points[i].cpu()
    area = sample_areas[i]
    predicted_area = predicted_sample_areas[i]
    center = torch.mean(points, axis=0)
    ax.scatter(points[..., 0], points[..., 1])
    ax.add_patch(plt.Polygon(points))
    ax.annotate("Area: %.2f" % area, center)
    ax.annotate("Predicted area: %.2f" % predicted_area, center + torch.tensor([0, -0.1]))
    print(area, predicted_area, flush = True)
fig.show()