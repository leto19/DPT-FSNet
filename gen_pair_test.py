import numpy as np
import librosa
import glob
import os
import h5py
import time


def gen_pair():

    test_clean_path = '/mnt/parscratch/users/acp20glc/VoiceBank/clean_testset_wav_16k'
    test_noisy_path = '/mnt/parscratch/users/acp20glc/VoiceBank/noisy_testset_wav_16k'
    test_mix_path = './dataset/voice_bank_mix/testset'

    test_clean_name = sorted(os.listdir(test_clean_path))
    test_noisy_name = sorted(os.listdir(test_noisy_path))

   # print(test_clean_name)
    #print(test_noisy_name)

    for count in range(len(test_clean_name)):

        clean_name = test_clean_name[count]
        noisy_name = test_noisy_name[count]
        #print(clean_name, noisy_name)
        if clean_name == noisy_name:
            file_name = '%s_%d' % ('test_mix', count+1)
            test_writer = h5py.File(test_mix_path + '/' + file_name, 'w')

            clean_audio, sr = librosa.load(os.path.join(test_clean_path, clean_name), sr=16000)
            noisy_audio, sr1 = librosa.load(os.path.join(test_noisy_path, noisy_name), sr=16000)

            test_writer.create_dataset('noisy_raw', data=noisy_audio.astype(np.float32), chunks=True)
            test_writer.create_dataset('clean_raw', data=clean_audio.astype(np.float32), chunks=True)
            test_writer.close()
        else:
            raise TypeError('clean file and noisy file do not match')

    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    test_file_list = sorted(glob.glob(os.path.join(test_mix_path, '*')))
    read_train = open("test_file_list", "w+")

    for i in range(len(test_file_list)):
        read_train.write("%s\n" % (test_file_list[i]))

    read_train.close()
    print('making test data finished!')


if __name__ == "__main__":
    gen_pair()
 