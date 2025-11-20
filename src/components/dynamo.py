import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import NoCredentialsError, ClientError
import pandas as pd
import io
import traceback
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from decimal import Decimal
import streamlit as st

class DynamoDBManager:
    """DynamoDB 관리 클래스"""
    
    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, 
                 region_name: str, conversations_table: str, messages_table: str):
        """
        DynamoDB 매니저 초기화
        
        Args:
            aws_access_key_id: AWS 액세스 키
            aws_secret_access_key: AWS 시크릿 키
            region_name: AWS 리전
            conversations_table: 대화 테이블 이름
            messages_table: 메시지 테이블 이름
        """
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.conversations_table_name = conversations_table
        self.messages_table_name = messages_table
        
        # DynamoDB 리소스 초기화
        self.dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        # 테이블 참조
        self.conversations_table = self.dynamodb.Table(conversations_table)
        self.messages_table = self.dynamodb.Table(messages_table)
        
        # 테이블 초기화
        self.create_tables_if_not_exists()
    
    def create_tables_if_not_exists(self):
        """DynamoDB 테이블이 없으면 생성"""
        try:
            existing_tables = [table.name for table in self.dynamodb.tables.all()]
            print(f"Existing tables: {existing_tables}")
            
            # Conversations 테이블 (전처리 또는 데이터 분석용)
            if self.conversations_table_name not in existing_tables:
                try:
                    # 테이블 용도에 따른 태그 추가
                    table_purpose = "preprocessing" if "preprocessing" in self.conversations_table_name else "dataanalysis"
                    
                    print(f"Creating {table_purpose} conversations table: {self.conversations_table_name}")
                    
                    self.dynamodb.create_table(
                        TableName=self.conversations_table_name,
                        KeySchema=[
                            {'AttributeName': 'session_uuid', 'KeyType': 'HASH'}
                        ],
                        AttributeDefinitions=[
                            {'AttributeName': 'session_uuid', 'AttributeType': 'S'},
                            {'AttributeName': 'user_id', 'AttributeType': 'S'},
                            {'AttributeName': 'created_at', 'AttributeType': 'N'}
                        ],
                        GlobalSecondaryIndexes=[
                            {
                                'IndexName': 'user_id-created_at-index',
                                'KeySchema': [
                                    {'AttributeName': 'user_id', 'KeyType': 'HASH'},
                                    {'AttributeName': 'created_at', 'KeyType': 'RANGE'}
                                ],
                                'Projection': {'ProjectionType': 'ALL'}
                            }
                        ],
                        BillingMode='PAY_PER_REQUEST',
                        Tags=[
                            {
                                'Key': 'Purpose',
                                'Value': table_purpose
                            },
                            {
                                'Key': 'Environment',
                                'Value': 'development'
                            }
                        ]
                    )
                    
                    # 테이블이 ACTIVE 상태가 될 때까지 대기
                    waiter = self.dynamodb.meta.client.get_waiter('table_exists')
                    waiter.wait(TableName=self.conversations_table_name)
                    
                    # GSI가 ACTIVE 상태가 될 때까지 대기
                    waiter = self.dynamodb.meta.client.get_waiter('table_exists')
                    waiter.wait(
                        TableName=self.conversations_table_name,
                        WaiterConfig={'Delay': 5, 'MaxAttempts': 20}
                    )
                    
                    print(f"✅ Created {table_purpose} conversations table: {self.conversations_table_name}")
                    print(f"  - Purpose: {table_purpose}")
                    print(f"  - Table Name: {self.conversations_table_name}")
                    print(f"  - Region: {self.dynamodb.meta.client.meta.region_name}")
                    
                except ClientError as e:
                    if e.response['Error']['Code'] != 'ResourceInUseException':
                        print(f"❌ Error creating conversations table: {e}")
                        raise
                    else:
                        print(f"ℹ️ Conversations table already exists: {self.conversations_table_name}")
            else:
                print(f"ℹ️ Conversations table already exists: {self.conversations_table_name}")
                
        except Exception as e:
            print(f"❌ Error in create_tables_if_not_exists: {e}")
            raise
        
        # Messages 테이블 (전처리 또는 데이터 분석용)
        if self.messages_table_name not in existing_tables:
            try:
                # 테이블 용도에 따른 태그 추가
                table_purpose = "preprocessing" if "preprocessing" in self.messages_table_name else "dataanalysis"
                
                print(f"Creating {table_purpose} messages table: {self.messages_table_name}")
                
                self.dynamodb.create_table(
                    TableName=self.messages_table_name,
                    KeySchema=[
                        {'AttributeName': 'conversation_id', 'KeyType': 'HASH'},
                        {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'conversation_id', 'AttributeType': 'S'},
                        {'AttributeName': 'timestamp', 'AttributeType': 'N'}
                    ],
                    BillingMode='PAY_PER_REQUEST',
                    Tags=[
                        {
                            'Key': 'Purpose',
                            'Value': table_purpose
                        },
                        {
                            'Key': 'Environment',
                            'Value': 'development'
                        }
                    ]
                )
                
                # 테이블이 ACTIVE 상태가 될 때까지 대기
                waiter = self.dynamodb.meta.client.get_waiter('table_exists')
                waiter.wait(TableName=self.messages_table_name)
                
                print(f"✅ Created {table_purpose} messages table: {self.messages_table_name}")
                print(f"  - Purpose: {table_purpose}")
                print(f"  - Table Name: {self.messages_table_name}")
                print(f"  - Region: {self.dynamodb.meta.client.meta.region_name}")
                
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceInUseException':
                    print(f"❌ Error creating messages table: {e}")
                    raise
                else:
                    print(f"ℹ️ Messages table already exists: {self.messages_table_name}")
        else:
            print(f"ℹ️ Messages table already exists: {self.messages_table_name}")

class S3Manager:
    """S3 관리 클래스"""
    
    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, 
                 region_name: str, bucket_name: str):
        """
        S3 매니저 초기화
        
        Args:
            aws_access_key_id: AWS 액세스 키
            aws_secret_access_key: AWS 시크릿 키
            region_name: AWS 리전
            bucket_name: S3 버킷 이름
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
    
    def upload_to_s3(self, data, content_type: str, conversation_id: str, use_jpeg: bool = False) -> Optional[str]:
        """
        데이터를 S3에 업로드하고 파일 경로를 반환
        
        Args:
            data: 업로드할 데이터
            content_type: 'dataframe' 또는 'figure'
            conversation_id: 대화 ID (S3 루트 폴더로 사용)
            use_jpeg: figure의 경우 JPEG 사용 여부 (기본값: False = PNG 사용)
        
        Returns:
            str: S3 객체 키 또는 None (실패 시)
        """
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]  # UUID의 앞 8자리만 사용
            
            if content_type == 'dataframe':
                # CSV 파일로 저장
                file_extension = 'csv'
                object_name = f"{conversation_id}/dataframes/{timestamp}_{unique_id}.{file_extension}"
                
                buffer = io.StringIO()
                data.to_csv(buffer, index=False)
                buffer.seek(0)
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_name,
                    Body=buffer.getvalue(),
                    ContentType='text/csv',
                    Metadata={
                        'upload_timestamp': timestamp,
                        'content_type': 'dataframe',
                        'conversation_id': conversation_id
                    }
                )
                
            elif content_type == 'figure':
                # PNG 또는 JPEG 이미지로 저장
                if use_jpeg:
                    file_extension = 'jpg'
                    image_format = 'jpeg'
                    mime_type = 'image/jpeg'
                else:
                    file_extension = 'png'
                    image_format = 'png'
                    mime_type = 'image/png'
                
                object_name = f"{conversation_id}/figures/{timestamp}_{unique_id}.{file_extension}"
                
                buffer = io.BytesIO()
                # PNG는 투명도 지원, JPEG는 더 작은 파일 크기
                if use_jpeg:
                    # JPEG는 투명도를 지원하지 않으므로 흰색 배경 추가
                    data.savefig(buffer, format=image_format, dpi=100, 
                               bbox_inches='tight', facecolor='white', quality=85)
                else:
                    data.savefig(buffer, format=image_format, dpi=100, 
                               bbox_inches='tight', transparent=True)
                buffer.seek(0)
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_name,
                    Body=buffer,
                    ContentType=mime_type,
                    Metadata={
                        'upload_timestamp': timestamp,
                        'content_type': 'figure',
                        'image_format': image_format,
                        'conversation_id': conversation_id
                    }
                )
            else:
                return None
                
            return object_name
            
        except NoCredentialsError:
            st.error("AWS 자격 증명을 찾을 수 없습니다.")
            return None
        except Exception as e:
            st.error(f"S3 업로드 오류: {e}")
            traceback.print_exc()
            return None
    
    def download_from_s3(self, object_name: str, content_type: str):
        """
        S3에서 파일을 다운로드하여 객체로 변환
        
        Args:
            object_name: S3 객체 키
            content_type: 콘텐츠 타입 ('dataframe_s3' 또는 'figure_s3')
        
        Returns:
            DataFrame 또는 BytesIO 객체 또는 None (실패 시)
        """
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_name)
            body = obj['Body']
            
            if content_type == 'dataframe_s3':
                csv_string = body.read().decode('utf-8')
                return pd.read_csv(io.StringIO(csv_string))
            elif content_type == 'figure_s3':
                img_data = body.read()
                img_buffer = io.BytesIO(img_data)
                img_buffer.seek(0)
                return img_buffer
                
        except Exception as e:
            st.error(f"S3 다운로드 오류: {e}")
            traceback.print_exc()
            return None
    
    def get_file_info_from_s3(self, object_name: str) -> Dict[str, Any]:
        """
        S3 객체의 메타데이터 가져오기
        
        Args:
            object_name: S3 객체 키
        
        Returns:
            dict: 파일 정보 (크기, 콘텐츠 타입, 수정 시간, 메타데이터)
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=object_name)
            return {
                'size': response['ContentLength'],
                'content_type': response['ContentType'],
                'last_modified': response['LastModified'],
                'metadata': response.get('Metadata', {})
            }
        except Exception as e:
            return {}
    
    def delete_from_s3(self, object_name: str) -> bool:
        """
        S3에서 파일 삭제
        
        Args:
            object_name: S3 객체 키
        
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_name)
            return True
        except Exception as e:
            st.error(f"S3 삭제 오류: {e}")
            return False

class ConversationManager:
    """대화 관리 클래스"""
    
    def __init__(self, dynamo_manager: DynamoDBManager, conversations_table_name: str = None):
        """
        대화 매니저 초기화
        
        Args:
            dynamo_manager: DynamoDB 매니저 인스턴스
            conversations_table_name: 사용할 대화 테이블명 (None이면 기본 테이블 사용)
        """
        self.dynamo_manager = dynamo_manager
        self.conversations_table_name = conversations_table_name or dynamo_manager.conversations_table_name
        self.conversations_table = dynamo_manager.dynamodb.Table(self.conversations_table_name)
        self.messages_table = dynamo_manager.messages_table
    
    def get_or_create_conversation(self, session_uuid: str, user_id: Optional[str] = None) -> Optional[Dict]:
        """
        대화 세션 생성 또는 조회
        
        Args:
            session_uuid: 세션 UUID
            user_id: 사용자 ID
        
        Returns:
            dict: 대화 정보 또는 None (실패 시)
        """
        try:
            response = self.conversations_table.get_item(Key={'session_uuid': session_uuid})
            
            if 'Item' in response:
                return response['Item']
            else:
                # 새 대화 생성
                now = datetime.now(timezone.utc)
                conversation_item = {
                    'session_uuid': session_uuid,
                    'user_id': user_id or 'anonymous',
                    'topic': '새 대화',
                    'created_at': self._datetime_to_timestamp(now),
                    'updated_at': self._datetime_to_timestamp(now)
                }
                self.conversations_table.put_item(Item=conversation_item)
                return conversation_item
        except Exception as e:
            st.error(f"대화 생성/조회 오류: {e}")
            return None
    
    def get_user_conversations(self, user_id: str) -> List[Dict]:
        """
        사용자의 모든 대화 조회
        
        Args:
            user_id: 사용자 ID
        
        Returns:
            list: 대화 목록
        """
        try:
            print(f"🔍 Querying conversations for user: {user_id}")
            print(f"📊 Using table: {self.conversations_table_name}")
            
            # 테이블 상태 확인
            try:
                table_status = self.conversations_table.meta.client.describe_table(
                    TableName=self.conversations_table_name
                )
                print(f"📋 Table status: {table_status['Table']['TableStatus']}")
                
                # GSI 상태 확인
                if 'GlobalSecondaryIndexes' in table_status['Table']:
                    for gsi in table_status['Table']['GlobalSecondaryIndexes']:
                        print(f"🔗 GSI {gsi['IndexName']}: {gsi['IndexStatus']}")
                
            except Exception as status_error:
                print(f"⚠️ Could not check table status: {status_error}")
            
            response = self.conversations_table.query(
                IndexName='user_id-created_at-index',
                KeyConditionExpression=Key('user_id').eq(user_id),
                ScanIndexForward=False  # 최신순 정렬
            )
            
            conversations = response.get('Items', [])
            print(f"✅ Found {len(conversations)} conversations for user {user_id}")
            return conversations
            
        except Exception as e:
            print(f"❌ Error querying conversations: {e}")
            print(f"🔍 Table: {self.conversations_table_name}")
            print(f"👤 User ID: {user_id}")
            
            # 테이블이 존재하지 않는 경우 빈 리스트 반환
            if "ResourceNotFoundException" in str(e):
                print(f"⚠️ Table {self.conversations_table_name} does not exist")
                return []
            
            st.error(f"대화 목록 조회 오류: {e}")
            return []
    
    def update_conversation_topic(self, session_uuid: str, topic: str):
        """
        대화 주제 업데이트
        
        Args:
            session_uuid: 세션 UUID
            topic: 새로운 주제
        """
        try:
            self.conversations_table.update_item(
                Key={'session_uuid': session_uuid},
                UpdateExpression='SET topic = :topic, updated_at = :now',
                ExpressionAttributeValues={
                    ':topic': topic,
                    ':now': self._datetime_to_timestamp(datetime.now(timezone.utc))
                }
            )
        except Exception as e:
            st.error(f"주제 업데이트 오류: {e}")
    
    def delete_conversation(self, session_uuid: str, s3_manager: S3Manager) -> bool:
        """
        대화 및 관련 메시지 삭제
        
        Args:
            session_uuid: 세션 UUID
            s3_manager: S3 매니저 인스턴스
        
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            # 메시지 먼저 삭제 (S3 파일 포함)
            msg_response = self.messages_table.query(
                KeyConditionExpression=Key('conversation_id').eq(session_uuid)
            )
            
            for msg in msg_response.get('Items', []):
                # S3 파일 삭제
                if msg.get('content_type') in ['dataframe_s3', 'figure_s3']:
                    try:
                        s3_manager.delete_from_s3(msg['content'])
                    except:
                        pass
                
                # 메시지 삭제
                self.messages_table.delete_item(
                    Key={
                        'conversation_id': session_uuid,
                        'timestamp': msg['timestamp']
                    }
                )
            
            # 대화 삭제
            self.conversations_table.delete_item(Key={'session_uuid': session_uuid})
            return True
        except Exception as e:
            st.error(f"대화 삭제 오류: {e}")
            return False
    
    def _datetime_to_timestamp(self, dt: datetime) -> Decimal:
        """datetime을 DynamoDB timestamp(Decimal)로 변환"""
        return Decimal(str(dt.timestamp()))

class MessageManager:
    """메시지 관리 클래스"""
    
    def __init__(self, dynamo_manager: DynamoDBManager):
        """
        메시지 매니저 초기화
        
        Args:
            dynamo_manager: DynamoDB 매니저 인스턴스
        """
        self.dynamo_manager = dynamo_manager
        self.conversations_table = dynamo_manager.conversations_table
        self.messages_table = dynamo_manager.messages_table
    
    def load_messages(self, conversation_id: str, table_name: str = None) -> List[Dict]:
        """
        대화 메시지 로드
        
        Args:
            conversation_id: 대화 ID
            table_name: 사용할 메시지 테이블명 (None이면 기본 테이블 사용)
        
        Returns:
            list: 메시지 목록
        """
        try:
            # 테이블명이 지정된 경우 해당 테이블 사용, 아니면 기본 테이블 사용
            if table_name:
                # DynamoDB 리소스에서 직접 테이블 참조
                table = self.dynamo_manager.dynamodb.Table(table_name)
            else:
                table = self.messages_table
            
            response = table.query(
                KeyConditionExpression=Key('conversation_id').eq(conversation_id),
                ScanIndexForward=True  # 시간순 정렬
            )
            
            messages = []
            for item in response.get('Items', []):
                messages.append({
                    'role': item['role'],
                    'content_type': item['content_type'],
                    'content': item['content'],
                    'timestamp': self._timestamp_to_datetime(item['timestamp'])
                })
            return messages
        except Exception as e:
            st.error(f"메시지 로드 오류: {e}")
            return []
    
    def save_message(
        self,
        conversation_id: str,
        role: str,
        content_type: str,
        content: str,
        llm_model: str = None,
        turn: Optional[int] = None,
    ) -> bool:
        """
        메시지 저장
        
        Args:
            conversation_id: 대화 ID
            role: 메시지 역할 (user/assistant)
            content_type: 콘텐츠 타입
            content: 메시지 내용
            llm_model: 사용된 LLM 모델명 (assistant 역할일 때만 사용)
            turn: 유저-에이전트 1턴 번호 (옵션)
        
        Returns:
            bool: 저장 성공 여부
        """
        try:
            now = datetime.now(timezone.utc)
            message_item = {
                'conversation_id': conversation_id,
                'timestamp': self._datetime_to_timestamp(now),
                'message_id': str(uuid.uuid4()),
                'role': role,
                'content_type': content_type,
                'content': content
            }
            
            # 턴 번호가 있는 경우 함께 저장
            if turn is not None:
                message_item['turn'] = turn
            
            # AI 답변인 경우 모델 정보 추가
            if role == 'assistant' and llm_model:
                message_item['LLM_model'] = llm_model
                
            self.messages_table.put_item(Item=message_item)
            
            # 대화 업데이트 시간 갱신
            self.conversations_table.update_item(
                Key={'session_uuid': conversation_id},
                UpdateExpression='SET updated_at = :now',
                ExpressionAttributeValues={
                    ':now': self._datetime_to_timestamp(now)
                }
            )
            
            # 첫 사용자 메시지인 경우 토픽 자동 설정
            if role == "user" and content_type in (
                "text",
                "learning_stage1_user_text",
                "learning_stage2_user_text",
                "learning_stage3_user_text",
            ):
                msg_count = self.messages_table.query(
                    KeyConditionExpression=Key('conversation_id').eq(conversation_id),
                    FilterExpression=Attr('role').eq('user'),
                    Select='COUNT'
                )
                if msg_count.get('Count', 0) == 1:
                    topic = content[:50] + ("..." if len(content) > 50 else "")
                    # ConversationManager 인스턴스가 필요하므로 여기서는 직접 업데이트
                    self.conversations_table.update_item(
                        Key={'session_uuid': conversation_id},
                        UpdateExpression='SET topic = :topic',
                        ExpressionAttributeValues={
                            ':topic': topic
                        }
                    )
            
            return True
        except Exception as e:
            st.error(f"메시지 저장 오류: {e}")
            traceback.print_exc()
            return False
    
    def _datetime_to_timestamp(self, dt: datetime) -> Decimal:
        """datetime을 DynamoDB timestamp(Decimal)로 변환"""
        return Decimal(str(dt.timestamp()))
    
    def _timestamp_to_datetime(self, timestamp: Decimal) -> datetime:
        """DynamoDB timestamp를 datetime으로 변환"""
        return datetime.fromtimestamp(float(timestamp))

class AgencyConversationManager:
    """Agency 대화 관리 클래스 - output, intent, assessment 저장용"""
    
    def __init__(self, dynamo_manager: DynamoDBManager, agency_table_name: str = None):
        """
        Agency 대화 매니저 초기화
        
        Args:
            dynamo_manager: DynamoDB 매니저 인스턴스
            agency_table_name: Agency 메시지 테이블명 (None이면 기본값 사용)
        """
        self.dynamo_manager = dynamo_manager
        self.agency_table_name = agency_table_name or "agency_messages"
        
        # Agency 테이블 참조
        self.agency_table = self.dynamo_manager.dynamodb.Table(self.agency_table_name)
        
        # 테이블 초기화
        self.create_agency_table_if_not_exists()
    
    def create_agency_table_if_not_exists(self):
        """Agency 대화 테이블이 없으면 생성"""
        existing_tables = [table.name for table in self.dynamo_manager.dynamodb.tables.all()]
        
        if self.agency_table_name not in existing_tables:
            try:
                # 테이블 용도에 따른 태그 추가
                if "preprocessing" in self.agency_table_name:
                    table_purpose = "preprocessing"
                elif "dataanalysis" in self.agency_table_name:
                    table_purpose = "dataanalysis"
                else:
                    table_purpose = "general"
                
                self.dynamo_manager.dynamodb.create_table(
                    TableName=self.agency_table_name,
                    KeySchema=[
                        {'AttributeName': 'conversation_id', 'KeyType': 'HASH'},
                        {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'conversation_id', 'AttributeType': 'S'},
                        {'AttributeName': 'timestamp', 'AttributeType': 'N'}
                    ],
                    BillingMode='PAY_PER_REQUEST',
                    Tags=[
                        {
                            'Key': 'Purpose',
                            'Value': table_purpose
                        },
                        {
                            'Key': 'Environment',
                            'Value': 'development'
                        }
                    ]
                )
                print(f"Created {table_purpose} agency table: {self.agency_table_name}")
                print(f"  - Purpose: {table_purpose}")
                print(f"  - Table Name: {self.agency_table_name}")
                print(f"  - Region: {self.dynamo_manager.dynamodb.meta.client.meta.region_name}")
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceInUseException':
                    raise
    
    def save_agency_response(
        self,
        conversation_id: str,
        output: str,
        intent: List,
        assessment: str,
        llm_model: str = 'gpt-4.1',
        turn: Optional[int] = None,
    ) -> bool:
        """
        Agency 응답 저장 (output, intent, assessment)
        
        Args:
            conversation_id: 대화 ID
            output: AI 응답 내용
            intent: 의도 분석 (List)
            assessment: 평가 결과 (JSON 문자열)
            llm_model: 사용된 LLM 모델명 (기본값: 'gpt-4.1')
        
        Returns:
            bool: 저장 성공 여부
        """
        try:
            now = datetime.now(timezone.utc)
            
            # assessment가 JSON 문자열인지 확인하고 파싱
            try:
                assessment_data = json.loads(assessment) if assessment else {}
                assessment_json = json.dumps(assessment_data, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                # JSON 파싱 실패 시 원본 문자열 그대로 저장
                assessment_json = assessment
            
            # intent에 Enum(TutorAct) 등이 포함될 수 있으므로 JSON 직렬화 가능한 형태로 변환
            def serialize_intent(value):
                try:
                    from enum import Enum
                    if isinstance(value, Enum):
                        return value.name
                    if isinstance(value, list):
                        return [serialize_intent(v) for v in value]
                    if isinstance(value, dict):
                        return {k: serialize_intent(v) for k, v in value.items()}
                    return value
                except Exception:
                    return str(value)

            safe_intent = serialize_intent(intent)

            agency_item = {
                'conversation_id': conversation_id,
                'timestamp': self._datetime_to_timestamp(now),
                'message_id': str(uuid.uuid4()),
                'role': 'assistant',
                'content': output,
                'intent': json.dumps(safe_intent, ensure_ascii=False),
                'assessment': assessment_json,
                'created_at': self._datetime_to_timestamp(now),
                'LLM_model': llm_model
            }
            if turn is not None:
                agency_item['turn'] = turn
            self.agency_table.put_item(Item=agency_item)
            return True
        except Exception as e:
            st.error(f"Agency 응답 저장 오류: {e}")
            traceback.print_exc()
            return False
    
    def save_user_input(self, conversation_id: str, user_input: str, turn: Optional[int] = None) -> bool:
        """
        사용자 입력 저장
        
        Args:
            conversation_id: 대화 ID
            user_input: 사용자 입력 내용
        
        Returns:
            bool: 저장 성공 여부
        """
        try:
            now = datetime.now(timezone.utc)
            agency_item = {
                'conversation_id': conversation_id,
                'timestamp': self._datetime_to_timestamp(now),
                'message_id': str(uuid.uuid4()),
                'role': 'user',
                'content': user_input,
                'intent': '',
                'assessment': '',
                'created_at': self._datetime_to_timestamp(now),
                'LLM_model': ''
            }
            if turn is not None:
                agency_item['turn'] = turn
            self.agency_table.put_item(Item=agency_item)
            return True
        except Exception as e:
            st.error(f"사용자 입력 저장 오류: {e}")
            traceback.print_exc()
            return False
    
    def _datetime_to_timestamp(self, dt: datetime) -> Decimal:
        """datetime을 DynamoDB timestamp(Decimal)로 변환"""
        return Decimal(str(dt.timestamp()))

# 유틸리티 함수들
def datetime_to_timestamp(dt: datetime) -> Decimal:
    """datetime을 DynamoDB timestamp(Decimal)로 변환"""
    return Decimal(str(dt.timestamp()))

def timestamp_to_datetime(timestamp: Decimal) -> datetime:
    """DynamoDB timestamp를 datetime으로 변환"""
    return datetime.fromtimestamp(float(timestamp))
