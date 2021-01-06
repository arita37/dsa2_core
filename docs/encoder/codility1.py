import math

def solution(predicted, observed):
    n = len(predicted)
    
    RMSE = 0
    for i in range(n):
        RMSE += (predicted[i] - observed[i]) ** 2
    
    RMSE = math.sqrt(RMSE / float(n))
    
    return RMSE
    
print(solution([4, 25, 0.75, 11], [3, 21, -1.25, 13]))       #    2.500
print(solution([-111, 555, 0, 0, 123], [123, 0, 0, 0, 123])) # ~269.363
print(solution([1], [1]))                                    #    0.000
print(solution([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]))           #   ~3.317
print(solution([-1, -2, -3], [1, 2, 6]))                     #   ~5.802



