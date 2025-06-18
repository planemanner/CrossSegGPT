import argparse

"""
mlflow 를 통해 언제든 결과를 reproduce 하기 편하도록 configuration 들도 저장하고 등등...
"""
def main(args):
    """
    trigger 를 받아서 실행하는 정도...
    각 process 는 별개로 구현 
    """
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # all arguments are action triggers
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--visualization', action='store_true')
    parser.add_argument('--preprocessing', action='store_true')

    args = parser.parse_args()
