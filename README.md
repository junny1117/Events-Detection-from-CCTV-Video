# Events-Detection-from-CCTV-Video

## 개요
- CCTV 영상에서 특정 구역 내의 객체(사람, 자동차)를 실시간으로 검지하고 침입자 발생과 같은 이벤트를 자동으로 인지하는 시스템 구축.
- 객체 검지 기술(YOLOv5)을 활용하여 영상을 분석하고 발생한 이벤트를 실시간으로 웹을 통해 확인할 수 있도록 구현.
- 사용자는 웹 인터페이스를 통해 관심 구역을 설정하고 발생한 이벤트를 기록 및 관리 가능.

## 목적
- **실시간 객체 검지 및 이벤트 처리**: CCTV 영상에서 사람과 자동차를 실시간으로 검지하고, 특정 구역에서 발생하는 이벤트(침입자 발생 등)를 자동으로 인식하여 경고.
- **웹 기반 관리**: 이벤트 검지 및 기록을 웹 인터페이스를 통해 제공하여 사용자가 실시간으로 모니터링하고 데이터 관리 가능.

## 사용 도구/기술
- **Flask**: 프레임워크
- **YOLOv5**: 객체검지
- **OpenCV**: 비디오처리
- **SQLite**: 데이터베이스
- **Flask-SocketIO**: 실시간 알림 (웹 소켓 통신)
- **HTML, CSS**: 웹페이지 탬플릿

## 데이터베이스 설계
- 발생한 이벤트의 종류(침입자 발생 등), 신뢰도(Confidence), 발생 시간을 Event 테이블에 기록.
- 이벤트 발생 시마다 Event 테이블에 추가.
- 웹 페이지의 결과 조회 페이지에서 최신 순으로 이벤트 로그를 확인할 수 있으며, Flask-SocketIO를 통해 메인 페이지에 실시간으로 표시.

## 객체 검지 및 이벤트 처리
- YOLOv5 모델을 사용하여 CCTV 영상 또는 지정된 영상에서 객체(사람, 자동차)를 실시간으로 검지.
- **이벤트 인지 로직**:
  - **침입**: 사람이 설정된 구역에 들어올 경우 인지.
  - **불법 주차**: 차량이 설정된 구역에 들어올 경우 인지.
  - **배회**: 특정 영역에 사람이 일정 시간 이상 머무를 경우 인지.
- 구역 설정은 OpenCV의 `selectROI` 메소드를 이용해 마우스로 원하는 구역을 드래그하여 설정 가능.

## 웹 애플리케이션 구조
- Flask를 사용하여 웹페이지와 기타 기능을 제공. 주요 페이지는 다음과 같음:
  - **로그인 페이지**: 관리자 로그인 기능 제공. 로그인되지 않은 상태에서 다른 페이지에 접근하면 로그인 페이지로 리다이렉트됨.
  - **메인 페이지**: 실시간 CCTV 영상을 표시하고, 검지된 이벤트를 실시간으로 보여주며, 구역 설정 버튼 제공.
  - **결과 조회 페이지**: 데이터베이스에 기록된 이벤트를 최신 순으로 조회 가능.
  - **설정 페이지**: 제한구역 및 불법주차 구역 설정 가능.

## 주요 구현 흐름
![image](https://github.com/user-attachments/assets/06d24b35-fd87-44fe-ba15-0dce611c7176)
![image](https://github.com/user-attachments/assets/37d9f36d-3647-431d-aaea-a3507d8dae52)

## 파일 목록
### detection.py - 객체 검지, 이벤트 인지 수행
### video_stream.py - 비디오 스트림 처리 수행 
### events.py - 이벤트 처리, 저장, 사용자 알림 수행
### database.py - 데이터베이스와의 연결 설정, 이벤트 데이터 CRUD 수행
### app.py - Flask 파일, 다양한 기능들을 통합하고 웹을 통해 동작하도록 함
### login.html - 로그인 페이지 템플릿
### index.html - 메인 페이지 템플릿
### events.html - 결과 조회 페이지 템플릿
### best.pt - YOLO 객체 검지 모델
### test.mp4 - 테스트 용 영상
### requirements.txt - 실행에 필요한 패키지 목록

## 실행 방법(리눅스에서만 가능)
1. 프로젝트 클론: `git clone https://github.com/junny1117/Events-Detection-from-CCTV-Video`
2. 필요한 패키지 설치: `pip install -r requirements.txt`
3. Flask 서버 실행: `flask run`
4. 브라우저에서 `127.0.0.1:5000`으로 접속

## 실행 결과 이미지
![image](https://github.com/user-attachments/assets/b2daba03-e1b1-4e2e-b719-f178453c172e)
![image](https://github.com/user-attachments/assets/bc5bf651-99d7-404f-9315-ed72ce1d7362)
![image](https://github.com/user-attachments/assets/1c6dc02d-1f9e-412f-b850-3247f98023c9)
