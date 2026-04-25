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

normalization_methods = ["none", "max_norm", "palm_base", "nose_base"]
print("Available normalization methods:")
for i, method in enumerate(normalization_methods, 1):
    print(f"{i}. {method}")
normalization_choice = int(input("Pilih metode normalisasi (1-4): ")) - 1
normalization_method = normalization_methods[normalization_choice]

available_features = ["hand_face_distance"]
print("Available features:")
for i, feature in enumerate(available_features, 1):
    print(f"{i}. {feature}")

selected_features = []
while True:
    feature_choice = int(input("Pilih fitur (nomor fitur, 0 untuk selesai): "))
    if feature_choice == 0:
        break
    if 1 <= feature_choice <= len(available_features):
        feature = available_features[feature_choice - 1]
        if feature not in selected_features:
            selected_features.append(feature)
        else:
            print(f"{feature} sudah dipilih")
    else:
        print("Pilihan tidak valid")

data = {
    'name' : model_name,
    'path' : model_path,
    'training_data' : model_training_data,
    'min_face' : min_face_req,
    'min_hand' : min_hand_req,
    'face_detection_type' : landmarkordetector,
    'labels' : labels,
    'normalization' : normalization_method,
    'features' : selected_features
}

with open(f"{model_name}.yml", "w") as f:
    f.write(yaml.dump(data))