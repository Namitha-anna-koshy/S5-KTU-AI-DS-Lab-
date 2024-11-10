from sklearn import datasets

# List of all dataset loaders in sklearn.datasets
dataset_loaders = [func for func in dir(datasets) if func.startswith("load_")]

# Display the dataset loaders
print("Available built-in datasets in sklearn:")
for loader in dataset_loaders:
    print(f"- {loader}")
