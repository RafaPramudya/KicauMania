import yaml

model_name = input("Masukan nama model: ")
model_path = f"{model_name}.keras"
model_training_data = f"{model_name}.trdata.csv"
min_face_req = int(input("Masukan minimal deteksi muka : "))
min_hand_req = int(input("Masukan minimal deteksi tangan : "))
landmarkordetector = None

if min_face_req > 0:
    landmarkordetector = "Landmark" if input("Landmark atau (deteksi saja) muka (Y untuk landmark, (bebas) untuk ): ") == 'Y' else "Detection"

label_count = int(input("Jumlah label : "))
labels = []

for i in range(label_count):
    labels.append(input(": "))

data = {
    'name' : model_name,
    'path' : model_path,
    'training_data' : model_training_data,
    'min_face' : min_face_req,
    'min_hand' : min_hand_req,
    'face_detection_type' : landmarkordetector,
    'num_label' : label_count,
    'labels' : labels
}

with open(f"{model_name}.yml", "w") as f:
    f.write(yaml.dump(data))