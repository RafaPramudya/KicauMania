import yaml

model_name = input("Masukan nama model: ")
model_path = f"{model_name}.keras"
model_training_data = f"{model_name}.trdata.csv"
min_face_req = int(input("Masukan minimal deteksi muka : "))
min_hand_req = int(input("Masukan minimal deteksi tangan : "))
landmarkordetector = None

if min_face_req > 0:
    landmarkordetector = "Landmark" if input("Landmark atau (deteksi saja) muka (Y untuk landmark, (bebas) untuk deteksi saja): ") == 'Y' else "Detection"

label_count = int(input("Jumlah label : "))
labels = []

for i in range(label_count):
    labels.append(input(": "))

normalization_methods = ["none", "max_norm", "palm_base", "nose_base", "norm_each"]
print("Available normalization methods:")
for i, method in enumerate(normalization_methods, 1):
    print(f"{i}. {method}")
normalization_choice = int(input(f"Pilih metode normalisasi (1-{len(normalization_methods)}): ")) - 1
normalization_method = normalization_methods[normalization_choice]

available_features = ["hand_face_distance"]
print("Available features:")
for i, feature in enumerate(available_features, 1):
    print(f"{i}. {feature}")

selected_features = []
while True:
    feature_choice = int(input("Pilih fitur (nomor fitur, 0 untuk selesai): "))
    if feature_choice == 0: break
    if 1 <= feature_choice <= len(available_features):
        feature = available_features[feature_choice - 1]
        if feature not in selected_features:
            selected_features.append(feature)
        else:
            print(f"{feature} sudah dipilih")
    else:
        print("Pilihan tidak valid")

print("-- Masukan arsitektur model --")
available_layers = ["relu", "sigmoid", "dropout"]
print("Available layers:")
for i, layer in enumerate(available_layers, 1):
    print(f"{i}. {layer}")

hidden_layers = []
while True:
    layer_choice = int(input("Masukan jenis layer (0 untuk selesai): "))
    if layer_choice == 0: break

    # Can be anything that the model is inputing
    
    if 1 <= layer_choice <= len(available_layers):
        layer_type = available_layers[layer_choice - 1]

        useful_value = input("Masukan jumlahnya (neuron, faktor, or anything useful): ")

        # Hardcoded value type casting
        if layer_type == "dropout" :    useful_value = float(useful_value)
        else:                           useful_value = int(useful_value)

        hidden_layers.append({f"{layer_type}" : useful_value})
    else: 
        print("Pilihan tidak valid")

print("Ringkasan Layer Model: ")
# print(" - input : dynamic")
print(hidden_layers)
# print(f" - softmax : {label_count}")

data = {
    'name' : model_name,
    'path' : model_path,
    'training_data' : model_training_data,
    'min_face' : min_face_req,
    'min_hand' : min_hand_req,
    'face_detection_type' : landmarkordetector,
    'labels' : labels,
    'normalization' : normalization_method,
    'features' : selected_features,
    'hidden_layer' : hidden_layers
}

with open(f"{model_name}.yml", "w") as f:
    f.write(yaml.dump(data))