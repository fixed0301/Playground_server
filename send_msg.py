from pyfcm import FCMNotification

APIKEY = "AIzaSyC3lbehMApXw5dEKp5_hOH-NdOHXeMEpGo"
TOKEN = "eBJZJJBkTJWJY46IA3CQeQ:APA91bHa-Ce_mKNX6zhp1kOH0L2vidUNYJspKsHatsqJ_0h-TRGWSiecfk4WI6Nsn5JvLRFHqRtaIa-dsOH3w0Z90Gnz4RzKzghgNd-1ZCDEQiwHIhKQtSj2MC7zzCQR0o8KW-QBu119"

# 파이어베이스 콘솔에서 얻어 온 서버 키를 넣어 줌
push_service = FCMNotification(APIKEY)


def sendMessage(body, title):
    # 메시지 (data 타입)
    data_message = {
        "body": body,
        "title": title
    }

    # 토큰값을 이용해 1명에게 푸시알림을 전송함
    result = push_service.single_device_data_message(registration_id=TOKEN, data_message=data_message)

    # 전송 결과 출력
    print(result)


sendMessage("배달의 민족", "치킨 8000원 쿠폰 도착!")