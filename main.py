from pathlib import Path
import numpy as np
import pandas as pd
from ART2 import ART2

file = Path('data') / "klastrowanie" / "cube.csv"
def main():
    data = pd.read_csv(file)
    train_data = data[["x","y","z"]].to_numpy()

    input_row_size = 3
    max_categories = 10
    rho = 0.20

    network = ART2(n=input_row_size, m=max_categories, rho=rho)
    network.compute(train_data)
    # # learn data array, row by row
    # for row in data:
    #     x1,x2,x3,y1 = row
    #     network.learn(np.array([x1,x2,x3]))

if __name__ == '__main__':
    main()