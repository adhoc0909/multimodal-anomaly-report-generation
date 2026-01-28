import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rc
import seaborn as sns


def set_korean_font(verbose: bool = False) -> bool:
    """한글 폰트 설정. Mac, Windows, Linux 지원. verbose=True면 메시지 출력."""
    system = platform.system()
    success = False
    font_name = None

    # OS별 한글 폰트 후보 목록
    font_candidates = {
        'Darwin': ['Arial Unicode MS', 'AppleGothic', 'Apple SD Gothic Neo'],
        'Windows': ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Dotum'],
        'Linux': ['Noto Sans CJK KR', 'NanumGothic', 'NanumBarunGothic', 'UnDotum', 'DejaVu Sans']
    }

    # 시스템에 설치된 폰트 목록
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    # 해당 OS의 폰트 후보에서 사용 가능한 폰트 찾기
    candidates = font_candidates.get(system, font_candidates['Linux'])

    for font in candidates:
        if font in available_fonts:
            font_name = font
            break

    if font_name:
        rc('font', family=font_name)
        success = True
        if verbose:
            print(f'Korean font set: {font_name} ({system})')
    else:
        if verbose:
            print(f'No Korean font found for {system}. Tried: {candidates}')
        success = False

    plt.rcParams['axes.unicode_minus'] = False
    return success


def count_plot(df, col, ax=None, figsize=(10, 6), palette="Blues_r", rotation=None, title=None, xlabel=None, ylabel=None, order='desc', orient='v', top_n=None, show=True):
    """
    order: 'desc'(내림차순), 'asc'(오름차순), None(정렬 안 함)
    orient: 'v'(세로, 기본값), 'h'(가로)
    top_n: 상위 N개만 표시 (None이면 전체)
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # 정렬 순서 결정
    if order == 'desc':
        order_list = df[col].value_counts().index.tolist()
    elif order == 'asc':
        order_list = df[col].value_counts(ascending=True).index.tolist()
    else:
        order_list = df[col].value_counts().index.tolist()

    # top_n 적용
    if top_n is not None:
        order_list = order_list[:top_n]
        df = df[df[col].isin(order_list)]

    # 방향에 따라 x/y 설정
    if orient == 'h':
        sns.countplot(data=df, y=col, palette=palette, ax=ax, order=order_list)
        ax.set_ylabel(ylabel if ylabel is not None else col)
        ax.set_xlabel(xlabel if xlabel is not None else 'Count')
    else:
        sns.countplot(data=df, x=col, palette=palette, ax=ax, order=order_list)
        ax.set_xlabel(xlabel if xlabel is not None else col)
        ax.set_ylabel(ylabel if ylabel is not None else 'Count')

    ax.set_title(title if title else f'{col} Count')
    if rotation:
        ax.tick_params(axis='x' if orient == 'v' else 'y', rotation=rotation)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def bar_plot(df, x_col, y_col, ax=None, figsize=(10, 6), hue=None, palette="Blues_r", rotation=None, title=None, xlabel=None, ylabel=None, show=True):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if hue:
        sns.barplot(x=x_col, y=y_col, hue=hue, data=df, palette=palette, ax=ax)
    else:
        sns.barplot(x=x_col, y=y_col, data=df, palette=palette, ax=ax)

    if rotation:
        ax.tick_params(axis='x', rotation=rotation)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def line_plot(df, x_col, y_col, ax=None, figsize=(12, 6), color=None, marker='o', linewidth=2, rotation=None, title=None, show=True):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    sns.lineplot(x=df[x_col], y=df[y_col], data=df, color=color, marker=marker, linewidth=linewidth, ax=ax)
    if rotation:
        ax.tick_params(axis='x', rotation=rotation)
    ax.set_title(title if title else f'{y_col} by {x_col} (Line Plot)')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def box_plot(df, col, hue=None, ax=None, figsize=(8, 6), palette=None, title=None, show=True):
    """
    단일 boxplot 또는 hue로 그룹 비교
    - hue: 그룹 비교할 컬럼명 (예: 'completed')
    - palette: 색상 리스트 (예: ['salmon', 'skyblue'])
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if palette is None:
        palette = ['skyblue', 'salmon']

    if hue:
        sns.boxplot(data=df, x=hue, y=col, palette=palette, ax=ax)
        ax.set_title(title if title else f'{col} by {hue}')
    else:
        sns.boxplot(y=df[col], color=palette[0], ax=ax)
        ax.set_title(title if title else f'{col} (Box Plot)')

    ax.set_ylabel(col)

    if show:
        plt.tight_layout()
        plt.show()
    return ax
