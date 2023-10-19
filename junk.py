import pickle

f = pickle.load(open("data/finetune_data_600_plus_url.pkl", "rb"))

print(f[0])