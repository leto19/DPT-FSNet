import numpy as np
import librosa
import glob
import os
import h5py
import time
import json

in_valid_json = json.load(open('/mnt/parscratch/users/acp20glc/VoiceBank/valid.json', 'r'))


def gen_pair():

    train_clean_path = '/mnt/parscratch/users/acp20glc/VoiceBank/clean_trainset_28spk_wav_16k'
    train_noisy_path = '/mnt/parscratch/users/acp20glc/VoiceBank/noisy_trainset_28spk_wav_16k'
    train_mix_path = './dataset/voice_bank_mix/trainset'
    valid_mix_path = './dataset/voice_bank_mix/validset'
    train_clean_name = sorted(os.listdir(train_clean_path))
    train_noisy_name = sorted(os.listdir(train_noisy_path))



    for count in range(len(train_clean_name)):

        clean_name = train_clean_name[count]
        noisy_name = train_noisy_name[count]
        #print(clean_name, noisy_name)
        if clean_name == noisy_name:
            #print(clean_name)
            if clean_name.replace(".wav","") in in_valid_json.keys():
                print("%s is in valid set" % clean_name.replace(".wav",""))
                file_name = '%s_%d' % ('valid_mix', count+1)
                valid_writer = h5py.File(valid_mix_path + '/' + file_name, 'w')

                clean_audio, sr = librosa.load(os.path.join(train_clean_path, clean_name), sr=16000)
                noisy_audio, sr1 = librosa.load(os.path.join(train_noisy_path, noisy_name), sr=16000)

                valid_writer.create_dataset('noisy_raw', data=noisy_audio.astype(np.float32), chunks=True)
                valid_writer.create_dataset('clean_raw', data=clean_audio.astype(np.float32), chunks=True)
                valid_writer.close()
            else:
                file_name = '%s_%d' % ('train_mix', count+1)
                train_writer = h5py.File(train_mix_path + '/' + file_name, 'w')

                clean_audio, sr = librosa.load(os.path.join(train_clean_path, clean_name), sr=16000)
                noisy_audio, sr1 = librosa.load(os.path.join(train_noisy_path, noisy_name), sr=16000)

                train_writer.create_dataset('noisy_raw', data=noisy_audio.astype(np.float32), chunks=True)
                train_writer.create_dataset('clean_raw', data=clean_audio.astype(np.float32), chunks=True)
                train_writer.close()
        else:
            raise TypeError('clean file and noisy file do not match')

    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    train_file_list = sorted(glob.glob(os.path.join(train_mix_path, '*')))
    read_train = open("train_file_list", "w+")

    for i in range(len(train_file_list)):
        read_train.write("%s\n" % (train_file_list[i]))

    read_train.close()
    print('making training data finished!')

    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    valid_file_list = sorted(glob.glob(os.path.join(valid_mix_path, '*')))
    read_valid = open("valid_file_list", "w+")

    for i in range(len(valid_file_list)):
        read_valid.write("%s\n" % (valid_file_list[i]))

    read_valid.close()
    print('making validation data finished!')



if __name__ == "__main__":
    gen_pair()
 