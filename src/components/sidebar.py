import time
import uuid
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.components.dynamo import ConversationManager, S3Manager, timestamp_to_datetime, MessageManager
from src.components.dataclass import MessageType


def load_messages_from_db(conversation_id: str, message_manager: MessageManager, s3_manager: S3Manager):
    """DB에서 메시지를 로드하여 session_state에 저장"""
    if not conversation_id or not message_manager:
        return False
    
    try:
        # DB에서 메시지 로드
        db_messages = message_manager.load_messages(conversation_id)
        
        if not db_messages:
            return False
        
        # 메시지를 세션 상태 형식으로 변환
        session_messages = []
        
        with st.spinner("대화 기록을 불러오는 중..."):
            for msg in db_messages:
                role = msg['role']
                content_type = msg['content_type']
                content = msg['content']
                
                # S3에 저장된 콘텐츠인 경우 복원
                if content_type in ['dataframe_s3', 'figure_s3']:
                    restored_content = s3_manager.download_from_s3(content, content_type)
                    if restored_content is not None:
                        if content_type == 'dataframe_s3':
                            # DataFrame 복원
                            session_messages.append([role, [[MessageType.DATAFRAME, restored_content]]])
                        elif content_type == 'figure_s3':
                            # Figure 복원 (BytesIO를 matplotlib Figure로 변환)
                            try:
                                restored_content.seek(0)
                                # PIL을 사용하여 이미지 로드 (더 안전함)
                                from PIL import Image
                                img = Image.open(restored_content)
                                
                                # matplotlib Figure로 변환
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.imshow(img)
                                ax.axis('off')
                                
                                session_messages.append([role, [[MessageType.FIGURE, fig]]])
                            except Exception as e:
                                print(f"Figure 복원 실패: {e}")
                                # Figure 복원 실패 시 텍스트로 대체
                                session_messages.append([role, [[MessageType.TEXT, f"차트 로드 실패: {e}"]]])
                    else:
                        # 복원 실패 시 에러 메시지
                        session_messages.append([role, [[MessageType.ERROR, f"콘텐츠 복원 실패: {content_type}"]]])
                else:
                    # 일반 텍스트/코드 메시지
                    if isinstance(content, str):
                        # role이 assistant(AI)인 경우에만 AI 응답 형식 감지 로직 적용
                        if role == "assistant":
                            # AI 응답 형식인지 확인 (핵심 내용, 상세 설명 포함 여부)
                            # 더 정확한 패턴 매칭을 위해 여러 패턴 확인
                            ai_response_patterns = [
                                '핵심 내용:', '상세 설명:', '**핵심 내용**', '**상세 설명**',
                                '출력 형식', '분석 결과', '결론', '요약'
                            ]
                            
                            is_ai_response = any(pattern in content for pattern in ai_response_patterns)
                            
                            if is_ai_response:
                                # AI 응답 형식이면 TEXT로 저장
                                session_messages.append([role, [[MessageType.TEXT, content]]])
                            else:
                                # AI 응답 형식이 아니면 CODE로 저장 (st.expander 안에서 st.code로 표시)
                                session_messages.append([role, [[MessageType.CODE, content]]])
                        else:
                            # user인 경우는 이전처럼 일반 텍스트로 저장
                            session_messages.append([role, [[MessageType.TEXT, content]]])
                    else:
                        # 문자열이 아닌 경우 TEXT로 저장
                        session_messages.append([role, [[MessageType.TEXT, content]]])
            
            # 세션 상태에 메시지 저장
            st.session_state["messages_SchoolFood"] = session_messages
            return True
        
    except Exception as e:
        st.error(f"메시지 로드 실패: {e}")
        return False


#--------------------------------#
#      Sidebar Configuration     #
#--------------------------------#
def render_model_setting():
    with st.sidebar:
        with st.expander("🧠 데이터 분석 모델 설정", expanded=False):

            model_option_general = st.selectbox(
                label="모델 선택지",
                options=["gpt-4.1-mini", "gpt-4o", "gpt-4.1"],
                index=0,
                help="데이터 분석을 위한 OpenAI 모델을 설정합니다."
            )

            if model_option_general != st.session_state["model_option_general"]:
                st.session_state["model_option_general"] = model_option_general
                st.success(f"모델 설정 완료: {model_option_general}")
                time.sleep(1)
                st.rerun()

def render_conversation_history(s3_manager: S3Manager, conversation_manager: ConversationManager, message_manager: MessageManager):
    with st.sidebar:
        st.divider()
        st.header("💬 대화 기록")
        
        # 새 대화 시작 버튼
        if st.button("➕ 새 대화 시작", use_container_width=True):
            st.session_state.session_uuid = str(uuid.uuid4())
            st.session_state.conversation_id = None
            st.session_state["messages_SchoolFood"] = []
            st.session_state["last_result_df"] = None
            st.session_state["last_figure"] = None
            st.success("🆕 새로운 대화를 시작합니다!")
            time.sleep(1)
            st.rerun()

        # 대화 목록 표시 (사용자 ID가 있는 경우)
        if hasattr(st.session_state, 'username'):
            try:
                conversations = conversation_manager.get_user_conversations(st.session_state.username)
                
                for convo in conversations:
                    # 현재 보고 있는 대화 강조
                    button_type = "primary" if convo['session_uuid'] == st.session_state.conversation_id else "secondary"
                    
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        if st.button(f"{convo['topic']} ({timestamp_to_datetime(convo['created_at']).strftime('%y-%m-%d %H:%M')})", 
                                    key=f"convo_{convo['session_uuid']}", use_container_width=True, type=button_type):
                            st.session_state.session_uuid = convo['session_uuid']
                            st.session_state.conversation_id = convo['session_uuid']
                            
                            # 이전 대화 메시지 로드
                            if load_messages_from_db(convo['session_uuid'], message_manager, s3_manager):
                                st.success(f"✅ '{convo['topic']}' 대화를 불러왔습니다!")
                                
                                # 에이전트 상태 복원 (데이터프레임이 있는 경우)
                                if st.session_state.get("messages_SchoolFood"):
                                    for role, content_list in st.session_state["messages_SchoolFood"]:
                                        for content in content_list:
                                            if isinstance(content, list) and len(content) == 2:
                                                message_type, message_content = content
                                                if message_type == MessageType.DATAFRAME and isinstance(message_content, pd.DataFrame):
                                                    # 마지막 데이터프레임을 현재 데이터로 설정
                                                    st.session_state["df_SchoolFood"] = message_content
                                                    if "python_tool" in st.session_state:
                                                        st.session_state["python_tool"].locals["df_SchoolFood"] = message_content
                                                        st.session_state["python_tool"].locals["df"] = message_content
                                                    break
                            else:
                                st.warning(f"⚠️ '{convo['topic']}' 대화 로드에 실패했습니다.")
                            
                            time.sleep(1)
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"del_{convo['session_uuid']}"):
                            conversation_manager.delete_conversation(convo['session_uuid'], s3_manager)
                            if st.session_state.conversation_id == convo['session_uuid']:
                                st.session_state.session_uuid = str(uuid.uuid4())
                                st.session_state.conversation_id = None
                            st.rerun()
            except Exception as e:
                st.error(f"대화 목록 로드 실패: {e}")


def render_data_info(uploaded_data: pd.DataFrame):
    with st.sidebar:
        st.markdown("---")

        # 현재 데이터 상태 표시
        if st.session_state.get("data_applied", False):
            if "uploaded_data" in st.session_state and st.session_state["uploaded_data"] is not None:
                st.caption(f"데이터 크기: {st.session_state['uploaded_data'].shape[0]}행 × {st.session_state['uploaded_data'].shape[1]}열")
                st.caption(f"👤 {st.session_state['username']}님 로그인됨")
        else:
            st.caption(f"데이터 크기: {uploaded_data.shape[0]}행 × {uploaded_data.shape[1]}열")
            st.caption(f"👤 {st.session_state['username']}님 로그인됨")
