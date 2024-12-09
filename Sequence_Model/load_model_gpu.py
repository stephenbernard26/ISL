import torch

class MultiModelInference:
    def __init__(self, models, device=None):
        """
        Initialize the class with a list of models and optionally specify the device.

        Args:
            models (list): List of PyTorch model instances.
            device (torch.device or str): The device to load models onto (e.g., "cuda" or "cpu").
        """
        self.models = models
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}

    def load_model_to_gpu(self, model_name, model):
        """
        Load a model onto the GPU and add it to the loaded models dictionary.

        Args:
            model_name (str): Name to identify the model.
            model (torch.nn.Module): PyTorch model instance.
        """
        model.to(self.device)
        model.eval()  # Set model to evaluation mode
        self.loaded_models[model_name] = model

    def unload_model_from_gpu(self, model_name):
        """
        Unload a model from the GPU to free up memory.

        Args:
            model_name (str): Name of the model to offload to CPU.
        """
        if model_name in self.loaded_models:
            self.loaded_models[model_name].to("cpu")
            del self.loaded_models[model_name]  # Remove model from loaded models

    def load_all_models(self):
        """
        Load all models in the list to the GPU.
        """
        for idx, model in enumerate(self.models):
            model_name = f"model_{idx}"
            self.load_model_to_gpu(model_name, model)

    def clear_gpu_memory(self):
        """
        Clear GPU memory by unloading all models.
        """
        for model_name in list(self.loaded_models.keys()):
            self.unload_model_from_gpu(model_name)
        torch.cuda.empty_cache()

    def inference(self, input_data, model_names=None):
        """
        Perform inference using specified models or all loaded models if none are specified.

        Args:
            input_data (torch.Tensor): Input data tensor to be used for inference.
            model_names (list or None): Names of models to use for inference. If None, uses all loaded models.

        Returns:
            dict: Dictionary with model names as keys and inference results as values.
        """
        results = {}
        input_data = input_data.to(self.device)  # Ensure input is on the GPU

        # Determine models to use for inference
        models_to_use = model_names or list(self.loaded_models.keys())
        
        with torch.no_grad():
            for model_name in models_to_use:
                if model_name in self.loaded_models:
                    results[model_name] = self.loaded_models[model_name](input_data)
                else:
                    print(f"Model '{model_name}' is not loaded on the GPU.")
        
        return results


import torch

# Function to clear GPU memory
def clear_gpu_memory():
    # Clear any loaded models or tensors from memory
    for obj in dir():
        del obj

    # Force garbage collection to free memory
    import gc
    gc.collect()
    
    # Clear any cached memory on the GPU
    torch.cuda.empty_cache()
    print("GPU memory has been cleared.")


if __name__ == '__main__':


    # Example usage
    clear_gpu_memory()