import pickle
import sys
import os




def fix_policy(e):
    e = e.replace("SRSF", "TIRESIAS")
    e = e.replace("MCS", "PCS")
    e = e.replace("_", "-")
    e = e.replace("perf", "JCT")
    e = e.replace("middle", "bal")
    e = e.replace("est", "pred")
    return e


def fix_figure(filename):
    fig_name = filename.split('figure')[-1].split('.pkl')[0]
    return fig_name


def _load_pickle(filename):
    data = list()
    with open(filename, "rb") as f:
        done = False
        while not done:
            try:
                obj = pickle.load(f)
                data.append(obj)
            except EOFError:
                done = True
    return data

def update_data(data, filename):
    with open(filename, 'wb') as fp:
        for d in data:
            pickle.dump(d, fp)

def main():

    filename = sys.argv[1]

    os.system(f'cp {filename} original_{filename}')

    data = _load_pickle(filename)


    for d in data:
        d['policy'] = fix_policy(d['policy'])
        d['figure'] = fix_figure(filename)
        
    update_data(data, filename)





if __name__ == '__main__':
    # main()
    data = _load_pickle(sys.argv[1])
    for d in data:
        print(d)
