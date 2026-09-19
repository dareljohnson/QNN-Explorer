import torch
from torch.utils.data import Dataset
# from torch_geometric.data import Data, Batch # If handling graph datasets

class TensorDatasetWrapper(Dataset):
    """Simple wrapper for tensor data (e.g., CSV features, direct quantum inputs)."""
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels
        if self.labels is not None:
            assert len(features) == len(labels), "Features and labels must have the same length"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

class ImageDatasetWrapper(Dataset):
    """Wrapper for a list/tensor of preprocessed images."""
    def __init__(self, image_tensors, labels=None):
        # Assuming image_tensors is a list of tensors or a single tensor [N, C, H, W]
        self.image_tensors = image_tensors
        self.labels = labels
        if self.labels is not None:
            assert len(image_tensors) == len(labels), "Images and labels must have the same length"

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.image_tensors[idx], self.labels[idx]
        else:
            return self.image_tensors[idx]

class TextDatasetWrapper(Dataset):
    """Wrapper for Hugging Face tokenizer outputs."""
    def __init__(self, encodings, labels=None):
        # encodings expected to be a dict {'input_ids': tensor, 'attention_mask': tensor}
        self.encodings = encodings
        self.labels = labels
        self.num_samples = len(encodings['input_ids'])
        if self.labels is not None:
            assert self.num_samples == len(labels), "Encodings and labels must have the same length"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            return item, self.labels[idx]
        else:
            return item

# class GraphDatasetWrapper(Dataset):
#     """Wrapper for torch_geometric graph data."""
#     def __init__(self, data_list):
#         # data_list is expected to be a list of torch_geometric.data.Data objects
#         self.data_list = data_list
#         # Note: Labels are often stored within the Data object (e.g., data.y)

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         # DataLoader with torch_geometric handles batching via Batch.from_data_list
#         return self.data_list[idx]

# Add other dataset types (Video) as needed 