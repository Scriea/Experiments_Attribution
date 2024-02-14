import pickle

a = pickle.load(open("ICL_Random_Wiki_2_Results.pkl", 'rb'))

for key, value in a.items():
    print(f"Q: {key}\nOutput{value}")