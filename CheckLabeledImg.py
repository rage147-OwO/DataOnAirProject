import os

def process_files(directory_path):
    txt_files = []
    jpg_files = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            txt_files.append(filename)
        elif filename.endswith('.jpg'):
            jpg_files.append(filename)

    for jpg_file in jpg_files:
        base_name = os.path.splitext(jpg_file)[0]
        corresponding_txt = base_name + '.txt'
        
        if corresponding_txt not in txt_files:
            jpg_file_path = os.path.join(directory_path, jpg_file)
            os.remove(jpg_file_path)
            print(f"Removed {jpg_file} as no corresponding .txt was found.")
        else:
            print(f"Keeping {jpg_file} as corresponding .txt exists.")

# Replace with the path to the directory containing your files
target_directory = r'D:\GitHub\DataOnAirTeamProject\data\team1\dataset_team1'

process_files(target_directory)

