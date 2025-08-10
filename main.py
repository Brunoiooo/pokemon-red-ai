import sys

sys.path.append('src') 

from Core import Core

def main():
    core = Core("PokemonRed")

    core.start()

if __name__ == "__main__":
    main()