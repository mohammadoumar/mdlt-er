import argparse

def main():
    argparser = argparse.ArgumentParser(description="mdlt-er: A tool for emotion recognition.")
    argparser.add_argument("--input", type=str, help="Path to the input data.")
    
    filename = argparser.parse_args().input
    print("Hello from mdlt-er! \nYou provided the input file: " + filename)


if __name__ == "__main__":
    main()
