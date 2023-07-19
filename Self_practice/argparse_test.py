import argparse

# argparser 선언
parser = argparse.ArgumentParser(description='test')

# 인수 추가
parser.add_argument("--name", type=str, help='first name only')
parser.add_argument("--age", type=int, default=20)

# 인수 분석
args = parser.parse_args()

# 가져오기
print(f"My name is {args.name} and {args.age} years old")