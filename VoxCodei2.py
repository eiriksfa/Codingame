import sys
import math

"""
Wait one round to check patterns, (initialize simulation)
Build Simulation, based on patterns
Find solution:
    Monte-Carlo Search
    Q-Learning
    Optmizitaion (Evolutionary algorithm)
"""

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# width: width of the firewall grid
# height: height of the firewall grid
width, height = [int(i) for i in input().split()]

# game loop
while True:
    # rounds: number of rounds left before the end of the game
    # bombs: number of bombs left
    rounds, bombs = [int(i) for i in input().split()]
    for i in range(height):
        map_row = input()  # one line of the firewall grid

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)

    print("3 0")
