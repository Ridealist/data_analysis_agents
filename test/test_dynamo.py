import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import json
import uuid
import streamlit as st


# 테이블과 버킷 설정
st.session_state["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
st.session_state["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
st.session_state["AWS_REGION_NAME"] = st.secrets["AWS_REGION_NAME"]
st.session_state["DYNAMODB_TABLE_NAME"] = st.secrets["DYNAMODB_TABLE_NAME"]
st.session_state["S3_BUCKET_NAME"] = st.secrets["S3_BUCKET_NAME"]


# DynamoDB와 S3 클라이언트 생성
try:
    dynamodb = boto3.resource(
        "dynamodb",
        aws_access_key_id=st.session_state["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.session_state["AWS_SECRET_ACCESS_KEY"],
        region_name=st.session_state["AWS_REGION_NAME"]
    )
    s3 = boto3.client(
        's3',
        aws_access_key_id=st.session_state["AWS_ACCESS_KEY_ID"] ,
        aws_secret_access_key=st.session_state["AWS_SECRET_ACCESS_KEY"],
        region_name=st.session_state["AWS_REGION_NAME"]
    )
    print("AWS 서비스 연결 성공!")
except (NoCredentialsError, PartialCredentialsError) as e:
    print("AWS 자격 증명 오류:", e)
    exit()

# 테이블 객체 가져오기
table = dynamodb.Table(st.session_state["DYNAMODB_TABLE_NAME"])

def upload_to_s3(content, content_type="text/plain"):
    """S3에 컨텐츠를 업로드하고 URL을 반환"""
    try:
        # 고유한 파일명 생성
        file_key = f"contents/{uuid.uuid4()}.txt"
        
        # S3에 업로드
        s3.put_object(
            Bucket=st.session_state["S3_BUCKET_NAME"],
            Key=file_key,
            Body=content.encode('utf-8'),
            ContentType=content_type
        )
        
        # S3 URL 생성
        url = f"s3://{st.session_state["S3_BUCKET_NAME"]}/{file_key}"
        return url
    except Exception as e:
        print("S3 업로드 오류:", e)
        return None

def insert_chat_data(student_id, timestamp, who, content, context):
    try:
        # 컨텐츠를 S3에 업로드
        content_url = upload_to_s3(content)
        if not content_url:
            raise Exception("S3 업로드 실패")

        # DynamoDB에 데이터 저장
        response = table.put_item(
            Item={
                "student_id": student_id,
                "timestamp": timestamp,
                "data": {
                    "who": who,
                    "content": content_url,  # S3 URL 저장
                    "context": context
                }
            }
        )
        print("데이터 삽입 성공:", response)
    except Exception as e:
        print("데이터 삽입 오류:", e)

# 데이터 조회 함수 (단일 항목 조회)
def get_chat_data(student_id, timestamp):
    try:
        response = table.get_item(
            Key={
                "student_id": student_id,
                "timestamp": timestamp
            }
        )
        if "Item" in response:
            return response["Item"]
        else:
            return None
    except Exception as e:
        print("데이터 조회 오류:", e)
        return None

# 예제 실행
if __name__ == "__main__":
    # 채팅 데이터 삽입 예제
    insert_chat_data(
        student_id="12345",
        timestamp="2024-02-22T12:00:00Z",
        who=["user", "agent"],
        content="이것은 테스트 대화 내용입니다.",
        context="code_review"
    )

    # 데이터 조회
    item = get_chat_data("12345", "2024-02-22T12:00:00Z")
    print("조회된 데이터:", item)
