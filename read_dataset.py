# Make sure to have datasets@main installed as this is a new feature
# pip install git+https://github.com/huggingface/datasets.git

from datasets import load_dataset, Video

dataset = load_dataset("pathto/yourdataset")
video_sample = dataset[0]["video"]
print(video_sample[0].shape) #Frame 0 shape