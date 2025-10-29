"""
LangGraph 워크플로우 시각화 스크립트

여러 방식으로 그래프를 시각화할 수 있습니다:
1. Mermaid 다이어그램 (텍스트)
2. ASCII 그래프 (텍스트)
3. PNG 이미지 (선택적 - 의존성 필요)
"""

import sys
from src.agency_langgraph.graph.workflow import create_agency_graph


def visualize_mermaid():
    """Mermaid 다이어그램으로 시각화 (텍스트)"""
    print("=" * 80)
    print("1. Mermaid 다이어그램 (복사해서 https://mermaid.live 에서 확인)")
    print("=" * 80)
    
    graph = create_agency_graph()
    mermaid_code = graph.get_graph().draw_mermaid()
    
    print(mermaid_code)
    print("\n")
    print("💡 위 코드를 복사해서 https://mermaid.live 에 붙여넣으면 시각화됩니다!")
    print()


def visualize_ascii():
    """ASCII 그래프로 시각화 (간단한 텍스트)"""
    print("=" * 80)
    print("2. ASCII 그래프 (터미널에서 바로 확인)")
    print("=" * 80)
    
    graph = create_agency_graph()
    
    try:
        ascii_graph = graph.get_graph().draw_ascii()
        print(ascii_graph)
    except Exception as e:
        print(f"ASCII 그래프 생성 실패: {e}")
        print("대신 노드와 엣지 정보를 출력합니다:\n")
        
        # 노드 정보
        nodes = graph.get_graph().nodes
        print("📍 노드 목록:")
        for node in nodes:
            print(f"  - {node}")
        
        print("\n🔗 엣지 정보:")
        edges = graph.get_graph().edges
        for edge in edges:
            print(f"  {edge}")
    
    print()


def visualize_png():
    """PNG 이미지로 저장 (pygraphviz 필요)"""
    print("=" * 80)
    print("3. PNG 이미지로 저장")
    print("=" * 80)
    
    graph = create_agency_graph()
    
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        
        # PNG 파일로 저장
        output_file = "agency_workflow_graph.png"
        with open(output_file, "wb") as f:
            f.write(png_data)
        
        print(f"✅ 그래프 이미지가 '{output_file}'에 저장되었습니다!")
        print(f"📂 위치: {output_file}")
        print()
        
    except ImportError as e:
        print("❌ PNG 생성을 위한 패키지가 설치되어 있지 않습니다.")
        print("\n설치 방법:")
        print("  pip install pygraphviz pillow")
        print("\n또는 Homebrew로 graphviz를 먼저 설치:")
        print("  brew install graphviz")
        print("  pip install pygraphviz pillow")
        print()
    except Exception as e:
        print(f"❌ PNG 생성 중 오류: {e}")
        print()


def print_graph_structure():
    """그래프 구조 상세 정보 출력"""
    print("=" * 80)
    print("4. 그래프 구조 상세 정보")
    print("=" * 80)
    
    graph = create_agency_graph()
    graph_obj = graph.get_graph()
    
    print("\n📋 워크플로우 요약:")
    print(f"  - 노드 개수: {len(graph_obj.nodes)}")
    print(f"  - 엣지 개수: {len(graph_obj.edges)}")
    
    print("\n📍 노드 목록:")
    for i, node in enumerate(graph_obj.nodes, 1):
        print(f"  {i}. {node}")
    
    print("\n🔗 엣지 연결:")
    for edge in graph_obj.edges:
        if hasattr(edge, 'source') and hasattr(edge, 'target'):
            print(f"  {edge.source} → {edge.target}")
        else:
            print(f"  {edge}")
    
    print("\n🎯 실행 흐름:")
    print("""
    START
      ↓
    assess (학생 상태 평가)
      ↓
    select (튜터 행동 선택)
      ↓
    generate (프롬프트 생성)
      ↓
    [조건부 분기]
      ├─→ respond (일반 응답)
      └─→ stream (스트리밍 응답)
      ↓
    END
    """)
    print()


def main():
    """메인 함수"""
    print("\n" + "🎨 " * 20)
    print("  LangGraph 워크플로우 시각화")
    print("🎨 " * 20 + "\n")
    
    # 1. Mermaid 다이어그램
    visualize_mermaid()
    
    # 2. ASCII 그래프
    visualize_ascii()
    
    # 3. 그래프 구조 정보
    print_graph_structure()
    
    # 4. PNG 이미지 (선택적)
    if len(sys.argv) > 1 and sys.argv[1] == "--png":
        visualize_png()
    else:
        print("=" * 80)
        print("💡 PNG 이미지로도 저장할 수 있습니다:")
        print("   python visualize_graph.py --png")
        print("=" * 80)
        print()
    
    print("\n✅ 시각화 완료!")
    print()
    print("🔗 유용한 링크:")
    print("  - Mermaid Live Editor: https://mermaid.live")
    print("  - LangGraph 문서: https://python.langchain.com/docs/langgraph")
    print()


if __name__ == "__main__":
    main()

