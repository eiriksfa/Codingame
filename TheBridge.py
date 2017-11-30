import sys
import math
import time

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

m = int(input())  # the amount of motorbikes to control
v = int(input())  # the minimum amount of motorbikes that must survive
l0 = input()  # L0 to L3 are lanes of the road. A dot character . represents a safe space, a zero 0 represents a hole in the road.
l1 = input()
l2 = input()
l3 = input()
lanes = [l0, l1, l2, l3]
print(l0, file=sys.stderr)
print(l1, file=sys.stderr)
print(l2, file=sys.stderr)
print(l3, file=sys.stderr)

moves = ['SPEED', 'SLOW', 'JUMP', 'WAIT', 'UP', 'DOWN']


def check_death(lanes, s, y, x):
    for i in range(s):
        if x-i >= len(lanes[0]):
            continue
        if lanes[y][x-i] == '0':
            return True


def update_state(lanes, s, states, move):
    if move == 'SPEED':
        s += 1
    elif move == 'SLOW':
        s -= 1
    nstates = []
    for (x, y, a) in states:
        if a == 0:
            continue
        x += s
        # TODO: What if limited by bridge rail?

        if move == 'UP':
            y -= 1
            if check_death(lanes, s, y, x) or check_death(lanes, s-1, y+1, x-1):
                a = 0
        elif move == 'DOWN':
            y += 1
            if check_death(lanes, s, y, x) or check_death(lanes, s-1, y-1, x-1):
                a = 0
        elif move == 'JUMP':
            if lanes[y][x] == '0':
                a = 0
        else:
            if check_death(lanes, s, y, x):
                a = 0
        nstates.append((x, y, a))
    return s, nstates


def check_status(lanes, states, v, m):
    dc = 0
    dx = 0
    for (x, y, a) in states:
        if a == 0:
            dc += 1
        else:
            dx = x
    if m - dc < v:
        return -1
    if dx >= len(lanes[0]):
        return 1
    return 0


# game loop
while True:
    s = int(input())  # the motorbikes' speed
    states = []
    for i in range(m):
        # x: x coordinate of the motorbike
        # y: y coordinate of the motorbike
        # a: indicates whether the motorbike is activated "1" or destroyed "0"
        x, y, a = [int(j) for j in input().split()]
        states.append((x, y, a))

    ns, nstate = update_state(lanes, s, states, 'SPEED')
    nstatus = check_status(lanes, nstate, v, m)

    print(nstatus, file=sys.stderr)

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)

    # A single line containing one of 6 keywords: SPEED, SLOW, JUMP, WAIT, UP, DOWN.
    print('SPEED')
