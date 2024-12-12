import os
import argparse
from collections import defaultdict

def count_classes(labels_dir):
    class_counts = {str(i): 0 for i in range(5)}
    empty_file_count = 0
    total_files = 0
    total_instances = 0

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    class_0_files = defaultdict(int)
    class_2_files = defaultdict(int)

    for label_file in label_files:
        total_files += 1
        file_path = os.path.join(labels_dir, label_file)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if not lines:
                empty_file_count += 1
            else:
                for line in lines:
                    class_id = line.split()[0]
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                        total_instances += 1
                        if class_id == '0':
                            class_0_files[label_file] += 1
                        elif class_id == '2':
                            class_2_files[label_file] += 1

    print("Top 10 files containing instances of class 0:")
    for file, count in sorted(class_0_files.items(), key=lambda item: item[1], reverse=True)[:10]:
        print(f"File {file} has {count} instances of class 0.")

    print("\nTop 10 files containing instances of class 2:")
    for file, count in sorted(class_2_files.items(), key=lambda item: item[1], reverse=True)[:10]:
        print(f"File {file} has {count} instances of class 2.")

    print("\nClass counts:")
    for class_id, count in class_counts.items():
        percentage = (count / total_instances) * 100 if total_instances > 0 else 0
        print(f"Class {class_id}: {count} ({percentage:.2f}%)")

    empty_percentage = (empty_file_count / total_files) * 100 if total_files > 0 else 0
    print(f"Empty files: {empty_file_count} ({empty_percentage:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count classes in label files.")
    parser.add_argument("labels_dir", help="Directory containing the label files.")
    
    args = parser.parse_args()
    
    count_classes(args.labels_dir)