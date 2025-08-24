import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import uuid
from datetime import datetime, timezone

from src.components.dynamo import DynamoDBManager, S3Manager, ConversationManager, MessageManager, timestamp_to_datetime

# --- 환경 설정 ---

# --- AWS S3 설정 ---
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = st.secrets.get("AWS_S3_BUCKET_NAME")
AWS_REGION = st.secrets.get("AWS_REGION", "ap-northeast-2")

# DynamoDB 테이블 이름
CONVERSATIONS_TABLE = st.secrets.get("DYNAMODB_CONVERSATIONS_TABLE", "chatbot_conversations")
MESSAGES_TABLE = st.secrets.get("DYNAMODB_MESSAGES_TABLE", "chatbot_messages")

# --- 매니저 클래스들 초기화 ---
try:
    # DynamoDB 매니저 초기화
    dynamo_manager = DynamoDBManager(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
        conversations_table=CONVERSATIONS_TABLE,
        messages_table=MESSAGES_TABLE
    )
    
    # S3 매니저 초기화
    s3_manager = S3Manager(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
        bucket_name=AWS_S3_BUCKET_NAME
    )
    
    # 대화 및 메시지 매니저 초기화
    conversation_manager = ConversationManager(dynamo_manager)
    message_manager = MessageManager(dynamo_manager)
    
except Exception as e:
    st.error(f"매니저 초기화 실패: {e}")

# --- Streamlit 애플리케이션 UI ---

st.set_page_config(page_title="고급 챗봇", layout="wide")
st.title("S3 연동 챗봇 (데이터프레임/그래프 저장)")

# 세션 상태 초기화
if "session_uuid" not in st.session_state:
    st.session_state.session_uuid = str(uuid.uuid4())

# secrets.toml에서 사용자 정보 가져오기
users_info = st.secrets["users"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "logged_in" not in st.session_state or st.session_state.logged_in == False:
    # 로그인 UI
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log in", type="primary"):
        if username not in users_info:
            st.error("Username을 확인해주세요.")
        else:
            if password != users_info[username]["password"]:
                st.error("Password를 확인해주세요.")
            else:
                # 로그인 성공 시 세션 상태 업데이트
                st.session_state.logged_in = True
                st.session_state["user_id"] = users_info[username]["userID"]
                st.session_state["username"] = username
                st.session_state["session_id"] = str(users_info[username]["userID"])
                
                st.success(f"로그인 성공! 환영합니다, {username}님!")
                time.sleep(0.5)
                st.rerun()


if st.session_state.logged_in:
    # 현재 대화 가져오기/생성
    current_conversation = conversation_manager.get_or_create_conversation(
        st.session_state.session_uuid,
        st.session_state.username
    )

    # 메시지 로드
    if current_conversation:
        if "messages" not in st.session_state or st.session_state.get("conversation_id") != current_conversation['session_uuid']:
            st.session_state.messages = message_manager.load_messages(current_conversation['session_uuid'])
            st.session_state.conversation_id = current_conversation['session_uuid']

    # 사이드바: 대화 기록
    with st.sidebar:
        st.header("대화 기록")
        if st.button("새 대화 시작"):
            st.session_state.session_uuid = str(uuid.uuid4())
            st.rerun()

        # 대화 목록
        conversations = conversation_manager.get_user_conversations(st.session_state.username)
        
        for convo in conversations:
            # 현재 보고 있는 대화 강조
            button_type = "primary" if convo['session_uuid'] == st.session_state.conversation_id else "secondary"
            
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(f"{convo['topic']} ({timestamp_to_datetime(convo['created_at']).strftime('%y-%m-%d %H:%M')})", 
                            key=f"convo_{convo['session_uuid']}", use_container_width=True, type=button_type):
                    st.session_state.session_uuid = convo['session_uuid']
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{convo['session_uuid']}"):
                    conversation_manager.delete_conversation(convo['session_uuid'], s3_manager)
                    if st.session_state.conversation_id == convo['session_uuid']:
                        st.session_state.session_uuid = str(uuid.uuid4())
                    st.rerun()

    # 메인 채팅 영역
    st.markdown("---")

    # 메시지 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content_type = msg["content_type"]
            content = msg["content"]
            
            if content_type == 'text':
                st.markdown(content)
            elif content_type == 'code':
                st.code(content, language='python')
                st.caption("💻 Python Code")
            elif content_type == 'log':
                with st.expander("🔍 Agent 내부 동작 로그", expanded=False):
                    try:
                        log_data = json.loads(content) if isinstance(content, str) else content
                        st.json(log_data)
                    except:
                        st.code(content, language='json')
            elif content_type == 'dataframe_s3':
                df = s3_manager.download_from_s3(content, content_type)
                if df is not None:
                    st.dataframe(df, use_container_width=True)
                    # 파일 확장자 추출하여 표시
                    file_ext = content.split('.')[-1] if '.' in content else 'csv'
                    st.caption(f"📊 Data from S3: `{content}` ({file_ext.upper()} format)")
            elif content_type == 'figure_s3':
                img_buffer = s3_manager.download_from_s3(content, content_type)
                if img_buffer is not None:
                    st.image(img_buffer)
                    # 파일 확장자 추출하여 표시
                    file_ext = content.split('.')[-1] if '.' in content else 'image'
                    st.caption(f"📈 Graph from S3: `{content}` ({file_ext.upper()} format)")

    # 채팅 입력
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 저장
        st.session_state.messages.append({
            "role": "user",
            "content_type": "text",
            "content": prompt,
            "timestamp": datetime.now(timezone.utc)
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        message_manager.save_message(st.session_state.conversation_id, "user", "text", prompt)
        
        # AI 응답 시뮬레이션
        with st.chat_message("assistant"):
            with st.spinner("AI가 생각 중입니다..."):
                
                # 데이터 분석 요청 감지
                if any(keyword in prompt.lower() for keyword in ['데이터', '분석', '그래프', 'data', 'chart', 'graph']):
                    
                    # 1. 내부 로그
                    log_content = {
                        "thought": "데이터 분석 요청 감지. 샘플 데이터 생성 및 시각화 수행",
                        "tool_used": "data_analysis",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "steps": [
                            "데이터프레임 생성 (CSV 형식)",
                            "S3 업로드",
                            "그래프 생성 (PNG 형식)",
                            "시각화 결과 반환"
                        ]
                    }
                    log_str = json.dumps(log_content, indent=2, ensure_ascii=False)
                    
                    message_manager.save_message(st.session_state.conversation_id, "assistant", "log", log_str)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content_type": "log",
                        "content": log_str,
                        "timestamp": datetime.now(timezone.utc)
                    })
                    
                    with st.expander("🔍 Agent 내부 동작 로그", expanded=False):
                        st.json(log_content)
                    
                    # 2. 데이터프레임 생성 및 S3 업로드 (CSV 형식)
                    df_sample = pd.DataFrame({
                        '제품': ['노트북', '마우스', '키보드', '모니터', '스피커'],
                        '판매량': [45, 120, 85, 32, 67],
                        '매출(만원)': [4500, 360, 425, 960, 335],
                        '이익률(%)': [15.2, 23.5, 18.7, 12.3, 20.1]
                    })
                    
                    s3_df_path = s3_manager.upload_to_s3(df_sample, 'dataframe', st.session_state.conversation_id)
                    if s3_df_path:
                        message_manager.save_message(st.session_state.conversation_id, "assistant", "dataframe_s3", s3_df_path)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content_type": "dataframe_s3",
                            "content": s3_df_path,
                            "timestamp": datetime.now(timezone.utc)
                        })
                        st.dataframe(df_sample, use_container_width=True)
                        st.caption(f"📊 Data saved to S3: `{s3_df_path}` (CSV format)")
                    
                    # 3. 그래프 생성 및 S3 업로드 (PNG 형식)
                    # PNG vs JPEG 선택 옵션 (기본: PNG)
                    use_jpeg = False  # True로 변경하면 JPEG 사용
                    
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # 막대 그래프 - 판매량
                    axes[0, 0].bar(df_sample['제품'], df_sample['판매량'], color='skyblue')
                    axes[0, 0].set_title('제품별 판매량')
                    axes[0, 0].set_xlabel('제품')
                    axes[0, 0].set_ylabel('판매량')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                    
                    # 파이 차트 - 매출 비중
                    axes[0, 1].pie(df_sample['매출(만원)'], labels=df_sample['제품'], autopct='%1.1f%%')
                    axes[0, 1].set_title('제품별 매출 비중')
                    
                    # 선 그래프 - 이익률
                    axes[1, 0].plot(df_sample['제품'], df_sample['이익률(%)'], marker='o', color='green')
                    axes[1, 0].set_title('제품별 이익률')
                    axes[1, 0].set_xlabel('제품')
                    axes[1, 0].set_ylabel('이익률(%)')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # 수평 막대 그래프 - 매출
                    axes[1, 1].barh(df_sample['제품'], df_sample['매출(만원)'], color='coral')
                    axes[1, 1].set_title('제품별 매출')
                    axes[1, 1].set_xlabel('매출(만원)')
                    axes[1, 0].set_ylabel('제품')
                    
                    plt.tight_layout()
                    
                    s3_fig_path = s3_manager.upload_to_s3(fig, 'figure', st.session_state.conversation_id, use_jpeg=use_jpeg)
                    if s3_fig_path:
                        message_manager.save_message(st.session_state.conversation_id, "assistant", "figure_s3", s3_fig_path)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content_type": "figure_s3",
                            "content": s3_fig_path,
                            "timestamp": datetime.now(timezone.utc)
                        })
                        st.pyplot(fig)
                        
                        # 파일 형식 표시
                        file_format = "JPEG" if use_jpeg else "PNG"
                        st.caption(f"📈 Graph saved to S3: `{s3_fig_path}` ({file_format} format)")
                    
                    plt.close()
                    
                    # 4. 텍스트 응답
                    response = f"""
                    데이터 분석을 완료했습니다! 📊
                    
                    **분석 결과 요약:**
                    - 가장 많이 판매된 제품: 마우스 (120개)
                    - 가장 높은 매출 제품: 노트북 (4,500만원)
                    - 가장 높은 이익률 제품: 마우스 (23.5%)
                    
                    **파일 저장 정보:**
                    - 데이터: CSV 형식으로 S3에 저장
                    - 그래프: {"JPEG" if use_jpeg else "PNG"} 형식으로 S3에 저장
                    
                    위 테이블과 차트를 통해 제품별 판매 현황을 확인하실 수 있습니다.
                    """
                elif any(keyword in prompt.lower() for keyword in ['코드', '프로그램', '함수', 'code', 'function', 'program']):
                    # 코드 생성 요청 감지
                    code_content = '''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(data_path):
    """
    데이터 분석 및 시각화 함수
    
    Args:
        data_path (str): 데이터 파일 경로
    
    Returns:
        dict: 분석 결과
    """
    # 데이터 로드
    df = pd.read_csv(data_path)
    
    # 기본 통계
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 히스토그램
    if df.select_dtypes(include=[np.number]).columns.any():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols[0]].hist(ax=axes[0,0], bins=20)
        axes[0,0].set_title(f'Histogram of {numeric_cols[0]}')
    
    # 상관관계 히트맵
    if len(df.select_dtypes(include=[np.number]).columns) > 1:
        sns.heatmap(df.corr(), annot=True, ax=axes[0,1])
        axes[0,1].set_title('Correlation Heatmap')
    
    # 박스플롯
    if df.select_dtypes(include=[np.number]).columns.any():
        df.boxplot(column=numeric_cols[0], ax=axes[1,0])
        axes[1,0].set_title(f'Boxplot of {numeric_cols[0]}')
    
    # 산점도
    if len(df.select_dtypes(include=[np.number]).columns) > 1:
        df.plot.scatter(x=numeric_cols[0], y=numeric_cols[1], ax=axes[1,1])
        axes[1,1].set_title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
    
    plt.tight_layout()
    plt.show()
    
    return stats

# 사용 예시
if __name__ == "__main__":
    result = analyze_data("your_data.csv")
    print("분석 결과:", result)'''
                    
                    # 코드 메시지 저장
                    message_manager.save_message(st.session_state.conversation_id, "assistant", "code", code_content)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content_type": "code",
                        "content": code_content,
                        "timestamp": datetime.now(timezone.utc)
                    })
                    
                    # 코드 표시
                    st.code(code_content, language='python')
                    st.caption("💻 Python Code")
                    
                    response = """
                    Python 코드를 생성했습니다! 🐍
                    
                    **코드 기능:**
                    - 데이터 로드 및 기본 통계 분석
                    - 다양한 시각화 (히스토그램, 히트맵, 박스플롯, 산점도)
                    - 재사용 가능한 함수 형태로 구성
                    
                    이 코드를 사용하여 데이터 분석을 수행할 수 있습니다.
                    """
                else:
                    # 일반 텍스트 응답
                    response = f"'{prompt}'에 대한 AI 응답입니다.\n\n실제 서비스에서는 OpenAI/Claude API를 연동하여 더 정교한 응답을 제공할 수 있습니다."
                
                # 응답 저장
                message_manager.save_message(st.session_state.conversation_id, "assistant", "text", response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content_type": "text",
                    "content": response,
                    "timestamp": datetime.now(timezone.utc)
                })
                st.markdown(response)

else:
    st.sidebar.error("로그인 실패!")
