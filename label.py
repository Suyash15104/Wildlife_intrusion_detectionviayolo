import os

def update_labels(label_dir):
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()

            if len(parts) > 0:
                parts[0] = "2"   # 🔥 change class ID to 2

            new_lines.append(" ".join(parts) + "\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

    print(f"Updated labels in {label_dir}")

# 🔥 Run for both folders
update_labels(r"C:\Users\91860\OneDrive\Desktop\project folder\elephant data\labels")