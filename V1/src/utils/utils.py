import numpy as np
import math


def reverse_schedule_sampling_exp(itr):
    #TODO: Parameters.
    """see predrnn-pytorch in GitHub."""

    if itr < 25000:
        r_eta, eta = 0.5, 0.5

    elif itr < 50000:
        r_eta = 1.0 - 0.5 * math.exp(-float((itr-25000)/5000))
        eta = 0.5 - (0.5/(25000)*(itr-25000))
    else:
        r_eta, eta = 1.0, 0.0

    r_random_flip = np.random.random_sample((2, 7))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample((2, 7))
    true_token = (random_flip < eta)

    ones = np.ones((112, 112, 1))
    zeros = np.zeros((112, 112, 1))

    real_input_flag = []

    for i in range(2):
        for j in range(14):
            if j < 7:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
                    
            else:
                if true_token[i, j - 7]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
    
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag, (2, 14, 112, 112, 1))
    return real_input_flag