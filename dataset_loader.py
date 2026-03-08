import os
from PIL import Image
from torch.utils.data import Dataset

class SketchPhotoDataset(Dataset):

    def __init__(self, sketch_dir, photo_dir):

        self.sketch_dir = sketch_dir
        self.photo_dir = photo_dir

        # only load jpg sketches
        self.files = [f for f in os.listdir(sketch_dir) if f.endswith(".jpg")]

        print("Sketch files found:", len(self.files))


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):

        sketch_file = self.files[idx]

        # convert sketch name to photo name
        photo_file = sketch_file.replace("-sz1", "")

        sketch_path = os.path.join(self.sketch_dir, sketch_file)
        photo_path = os.path.join(self.photo_dir, photo_file)

        sketch = Image.open(sketch_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")

        # temporary attribute (for your multimodal project)
        text_attribute = "female, black hair"

        return sketch, photo, text_attribute


dataset = SketchPhotoDataset(
    sketch_dir="sketches",
    photo_dir="photos"
)

print("Dataset loaded")
print("Total pairs:", len(dataset))

sketch, photo, text = dataset[0]

print("Sample text attribute:", text)