import sys
import math
import time

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

nbikes = int(input())  # the amount of motorbikes to control
nsurvive = int(input())  # the minimum amount of motorbikes that must survive
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
    if s < 0:
        s = 0
    nstates = []
    for (x, y, a) in states:
        if a == 0:
            continue
        x += s

        if move == 'UP':
            y -= 1
            if check_death(lanes, s, y, x) or check_death(lanes, s-1, y+1, x-1):
                a = 0
        elif move == 'DOWN':
            y += 1
            if check_death(lanes, s, y, x) or check_death(lanes, s-1, y-1, x-1):
                a = 0
        elif move == 'JUMP':
            if x < len(lanes) or lanes[y][x] == '0':
                a = 0
        else:
            if check_death(lanes, s, y, x):
                a = 0
        nstates.append((x, y, a))
    return s, nstates


def check_status(lanes, states, nsurvive, nbikes):
    dc = 0
    dx = 0
    for (x, y, a) in states:
        if a == 0:
            dc += 1
        else:
            dx = x
    if nbikes - dc < nsurvive:
        return -1
    if dx >= len(lanes[0]):
        return 1
    return 0


def check_bounds(states, move, s):
    if move == 'SLOW' and s < 1:
        return False
    if not (move == 'UP' or move == 'DOWN'):
        return True
    for (_, y, a) in states:
        if move == 'UP' and y == 0 and a == 1:
            return False
        if move == 'DOWN' and y == 3 and a == 1:
            return False
    return True


def rec_search(lanes, s, states, move, nsurvive, nbikes, depth):
    ns, nstates = update_state(lanes, s, states, move)
    nstatus = check_status(lanes, nstates, nsurvive, nbikes)
    if nstatus == -1 or nstatus == 1:
        return nstatus
    if depth > 30:
        return -1
    for move in moves:
        if not check_bounds(nstates, move, ns):
            continue
        nstatus = rec_search(lanes, ns, nstates, move, nsurvive, nbikes, depth+1)
        if nstatus == 1:
            return nstatus


def search(lanes, s, states, nsurvive, nbikes):
    for move in moves:
        if not check_bounds(states, move, s):
            continue
        nstatus = rec_search(lanes, s, states, move, nsurvive, nbikes, 0)
        if nstatus == 1:
            return move


# game loop
while True:
    s = int(input())  # the motorbikes' speed
    states = []
    for i in range(nbikes):
        # x: x coordinate of the motorbike
        # y: y coordinate of the motorbike
        # a: indicates whether the motorbike is activated "1" or destroyed "0"
        x, y, a = [int(j) for j in input().split()]
        states.append((x, y, a))

    move = search(lanes, s, states, nsurvive, nbikes)

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)

    # A single line containing one of 6 keywords: SPEED, SLOW, JUMP, WAIT, UP, DOWN.
    print(move)
