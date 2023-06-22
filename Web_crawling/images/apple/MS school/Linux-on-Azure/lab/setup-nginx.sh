# NGINX 설치 및 샘플 페이지 생성
# 패키지 소스 업데이트
sudo apt-get -y update

# NGINX 설치
sudo apt-get -y install nginx

# index.html 파일 만들기
fileName=/var/www/html/index.html
sudo sh -c "echo 'Running Sample Web from host $(hostname)'>${fileName}"