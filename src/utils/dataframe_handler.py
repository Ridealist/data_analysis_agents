import streamlit as st
import pandas as pd
import numpy as np

def safe_dataframe_display(df, max_rows=5, use_container_width=False):
    """
    PyArrow 변환 오류를 방지하는 안전한 데이터프레임 표시 함수
    """
    try:
        # 데이터프레임을 복사하고 안전한 타입으로 변환
        safe_df = df.copy()
        
        # object 타입 컬럼들을 문자열로 변환
        for col in safe_df.select_dtypes(include=['object']).columns:
            safe_df[col] = safe_df[col].astype(str)
        
        # datetime 타입 컬럼들을 문자열로 변환
        for col in safe_df.select_dtypes(include=['datetime64']).columns:
            safe_df[col] = safe_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 복잡한 numpy 타입들을 기본 타입으로 변환
        for col in safe_df.columns:
            if hasattr(safe_df[col].dtype, 'type'):
                try:
                    if safe_df[col].dtype.type in [np.int64, np.int32, np.int16, np.int8]:
                        safe_df[col] = safe_df[col].astype(int)
                    elif safe_df[col].dtype.type in [np.float64, np.float32]:
                        safe_df[col] = safe_df[col].astype(float)
                except:
                    safe_df[col] = safe_df[col].astype(str)
        
        return safe_df.head(max_rows)
    except Exception as e:
        st.warning(f"⚠️ 데이터프레임 표시 중 오류 발생: {str(e)}")
        # 오류 발생 시 기본 정보만 표시
        return pd.DataFrame({
            '컬럼명': df.columns.tolist(),
            '데이터 타입': [str(dtype) for dtype in df.dtypes],
            '행 수': [len(df)] * len(df.columns)
        })
