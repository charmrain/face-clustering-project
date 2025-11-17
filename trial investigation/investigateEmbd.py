import numpy as np
import pandas as pd

# load the .npz file
data = np.load(r"D:\project\photo\json\m5_collected_images\chine-Scene-490_frame_0001.faces.npz")

# see what’s inside
print(data.files)  # lists all keys, e.g. ['embedding', 'bbox', 'kps']

# access each item
embedding = data['embeddings']
bbox = data['bboxes']
score = data['scores']
image_path = data['image_path']

# inspect shapes or values
print(embedding.shape)
print(embedding[:5])

# Convert to DataFrame
df = pd.DataFrame(embedding)

# Save to CSV
# df.to_csv("embedding.csv", index=False)
