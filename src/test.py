# some analysis 
import os 

def print_exps( path = "../results/" ):
    exps = os.listdir(path)
    for exp in exps:
        if exp[:3] == "SVM" and exp[-9:] == "iter.xlsx":
            print(exp)

if __name__ == "__main__":
    print_exps()
