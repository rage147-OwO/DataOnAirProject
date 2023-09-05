import os

def modify_labels(file_path):
    with open(file_path, 'r+') as file:
        lines = file.readlines()
        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Modify the class to 0
                parts[0] = '0'
                modified_line = ' '.join(parts)
                modified_lines.append(modified_line)
        file.seek(0)
        file.truncate()
        for modified_line in modified_lines:
            file.write(modified_line + '\n')

def modify_labels_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            modify_labels(file_path)
            print(f"Modified {file_path}")



# Replace with the path to the directory containing your text files
target_directory = r'D:\GitHub\DataOnAirTeamProject\data\team1\dataset_team1'

modify_labels_in_directory(target_directory)
