import os
import pickle


utils_file_folder = os.path.dirname(os.path.realpath(__file__))


def get_encoder_decoder():
    data_folder = os.path.join(utils_file_folder, '..', 'shakespeare')
    meta_path = os.path.join(data_folder, 'meta.pkl')

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return encode, decode
