import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def overlapped_bar(df, show=False, width=0.75, alpha=.5, title='', xlabel='', ylabel='',
                   hide_xsticks=True, **plot_kwargs):
    """
    Like a stacked bar chart except bars on top of each other with transparency.
    :param df: data df
    :param show: show in ide
    :param width: bar width
    """
    plt.figure(figsize=(10, 6))  # 设置plt的尺寸
    xlabel = xlabel or df.index.name  # 标签
    N = len(df)   # 类别数
    M = len(df.columns)   # 列数
    indices = np.arange(N)
    colors = ['steelblue', 'firebrick', 'darksage', 'goldenrod', 'gray'] * int(M / 5. + 1)  # 颜色
    for i, label, color in zip(range(M), df.columns, colors):
        kwargs = plot_kwargs
        kwargs.update({'color': color, 'label': label})
        plt.bar(indices, df[label], width=width, alpha=alpha if i else 1, **kwargs)
        plt.xticks(indices + .5 * width, ['{}'.format(idx) for idx in df.index.values])
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if hide_xsticks:  # 如果水平坐标太多，隐藏水平坐标
        plt.xticks([])
    if show:
        plt.show()
    return plt.gcf()
# def draw_bars():
#     df = read_excel_to_df(os.path.join(DATA_DIR, "msa_all_counts.xlsx"))
#     df["ratio"] = df["all_sum"] / df["sum"]
#     df = df.sort_values(by=["ratio"], ascending=True)  # 从小到大排序
#     df_sum = df["sum"] / df["sum"]
#     df_all_sum = df["all_sum"] / df["sum"]
#     avg = round(np.average(df_all_sum), 4)
#     std = round(float(np.std(df_all_sum)), 4)
#     print(f"[Info] improve ratio: {avg}±{std}")   # 获取比例
#     low = df_sum   # 低区数值
#     high = df_all_sum  # 高区数值
#     df = pd.DataFrame(np.matrix([high, low]).T, columns=['Ours', 'AF2'])
#     overlapped_bar(df, xlabel="target", ylabel="times", show=True)
# if __name__ == '__main__':
#   draw_bars()