import numpy as np
import matplotlib.pyplot as plt
from bit_numbers import Numbers as N 

from functions import transform_functions as tf

import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

def change(num_of_rand :int, p :np.ndarray) -> np.ndarray:
    rand_idx = np.random.randint(0, 30, num_of_rand)
    p = p.copy().ravel()
    for idx in rand_idx:
        p[idx] = np.where(p[idx] == 1, -1, 1)

    return p

def get_matching_percentage(input:np.ndarray, target :np.ndarray) -> np.ndarray:
    matching_pixels = np.sum(input.ravel() == target.ravel())
    total_pixels = len(input.ravel())
    matching_percentage = (matching_pixels / total_pixels) * 100
    return matching_percentage

def main():
    _input  = np.array([N._0, N._1, N._2, N._3, N._4, N._5, N._6, N._7, N._8, N._9])
    _target = np.array([N._0, N._1, N._2, N._3, N._4, N._5, N._6, N._7, N._8, N._9])
    reshape_target = _target.reshape(10, 30)
    reshape_input = _input.reshape(10, 30)

    W = reshape_target.T @ np.linalg.pinv(reshape_input).T

    learning_rate = 0.01 # 학습률
    momentem = 1 # Update 시 이전 Weight Vector 값을 얼마나 반영할 것인지
    num_of_random = 3 # 학습 시 랜덤으로 변경할 픽셀갯수
    num_iterations = 1000 # 전체 반복 횟수
    check_range = 20 # error 안정 여부를 확인할 범위
    is_valid = False # error 안정 여부 : check_range만큼 연속으로 에러가 0.0 이면 True
    
    errors = [] 
    iteration = 0 # initialize

    while (iteration < num_iterations and is_valid == False):
        for idx, p in enumerate(_input):
            _p = change(num_of_random, p) 
            n = W @ _p.reshape(30, 1) 

            s_a = tf.hardlims(n)
            error = _target[idx].reshape(30, 1) - s_a

            error_norm = np.linalg.norm(error)
            errors.append(error_norm)

            W = momentem*W + learning_rate*(error @ _p.ravel().reshape(1, 30))

            if iteration >= num_iterations:
                break
            if len(errors) > check_range and all(value <= 0.01 for value in errors[iteration-check_range:iteration]):
                is_valid = True
                break
            iteration += 1
    
    # plot error graph
    plt.plot(list(range(0, len(errors))), errors, 'r', label="error norm")
    plt.title(f"Error graph of the number of iterations : {errors[-1]}")
    plt.legend()
    plt.show()

    # check the results
    for idx, p in enumerate(_input):
        _p = change(3, p)
        n = W @ _p.reshape(30, 1)
        a = tf.hardlims(n)

        matching_percentage = get_matching_percentage(input=a, target=_target[idx])
        plt.subplot(1, 3, 1)
        plt.title("Input (Noisy)")
        plt.imshow(_p.reshape(6, 5), cmap='binary')

        plt.subplot(1, 3, 2)
        plt.title(f"Result : {round(matching_percentage, 3)}")
        plt.imshow(a.reshape(6, 5), cmap='binary')

        plt.subplot(1, 3, 3)
        plt.title("Answer")
        plt.imshow(_target[idx], cmap='binary')
        plt.show()
    
if __name__ == "__main__":
    main()