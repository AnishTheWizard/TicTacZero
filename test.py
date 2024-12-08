import numpy as np

theta = np.array([0, 0, 0])
x = np.array([[1,1,0], [1,1,1], [1,0,1], [1,0,0]])
y = np.array([-1, 1, 1, -1])

finished = False

def error(y, x, theta, i):
  return (y[i] * (np.dot(x[i], theta))) <= 0

it = 0
while not finished:
  print("ITERATION", it,":", theta)
  finished = True
  it+=1
  for i in range(len(y)):
    print(error(y, x, theta, i))
    if error(y, x, theta, i):
      theta = theta + y[i] * x[i]
      finished = False
      
      
print("FINAL THETA", theta)