import json

in_valid_json = json.load(open('/mnt/parscratch/users/acp20glc/VoiceBank/valid.json', 'r'))

print(in_valid_json)


with open("train_file_list", "r") as f:
    train_file_list = f.readlines()

new_train_file_list = []
new_valid_file_list = []
for l in train_file_list:
    f_name = l.split('/')[-1].split('.')[0]
    print(f_name)
    if f_name in in_valid_json.keys():
        new_valid_file_list.append(l)
    else:
        new_train_file_list.append(l)

with open("train_file_list", "w") as f:
    f.writelines(new_train_file_list)

with open("valid_file_list", "w") as f:
    f.writelines(new_valid_file_list)
    