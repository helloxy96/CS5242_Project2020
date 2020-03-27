
import matplotlib.pyplot as plt
import sys
import numpy as np

def draw(A, B):

    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, len(A)+1), A)  # train loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train'], loc="upper left")
    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, len(B)+1), B)  # train accuracy (on epoch end)
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train'], loc="upper left")
    plt.show()

if __name__ == "__main__":
    file_A, file_B = sys.argv[1], sys.argv[2]

    A = np.load(file_A).flatten()
    B = np.load(file_B).flatten()

    print(A.shape)
    print(B.shape)
    
    draw(A,B)