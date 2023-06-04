from matplotlib import pyplot as plt
import json


def plot(cut_length, score_and_rank_dics, labels, args, out_name='tmp.png'):
    fig_size = 8
    font_size = 25
    plot_size = 20
    fig = plt.figure(figsize=(fig_size, fig_size))

    ax1 = fig.subplots()
    for i, score_and_rank_dic in enumerate(score_and_rank_dics):
        results = score_and_rank_dic['results']
        avg_word_ranks = score_and_rank_dic['avg_word_ranks']
        results = results[:-(cut_length - 1)]
        avg_word_ranks = avg_word_ranks[:-(cut_length - 1)]

        ax1.scatter(avg_word_ranks,
                    results,
                    s=plot_size,
                    label=labels[i]
                    )

    if args.emb_type == 'bert':
        plt.ylim(50, 75)
    elif args.emb_type == 'glove':
        plt.ylim(40, 60)
        ticks = [40, 45, 50, 55, 60]
        plt.yticks(ticks)

    x_label = "$R_{dataset}$"
    y_label = "Performance"
    margin = 0.15

    fig.subplots_adjust(left=margin, bottom=margin)
    # fig.subplots_adjust(left=0.1)
    lgnd = ax1.legend(fontsize=font_size-8)
    legend_marker_size = 120
    lgnd.legendHandles[0]._sizes = [legend_marker_size]
    lgnd.legendHandles[1]._sizes = [legend_marker_size]
    lgnd.legendHandles[2]._sizes = [legend_marker_size]
    lgnd.legendHandles[3]._sizes = [legend_marker_size]

    plt.rcParams["font.size"] = 18
    plt.tick_params(labelsize=font_size)
    plt.grid()
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.savefig(out_name, dpi=500)


def main():
    emb_type = args.emb_type
    org_dir = args.json_dir
    task = 'STS-14'
    paths = []
    if emb_type == 'glove':
        paths.append(f'{org_dir}/{task}_GloVe.json')
        paths.append(f'{org_dir}/{task}_GloVe-wh.json')
        paths.append(f'{org_dir}/{task}_GloVe-Fdeb.json')
        paths.append(f'{org_dir}/{task}_GloVe-Fdeb-wh.json')
        labels = ['GloVe', 'GloVe-wh', 'GloVe-Fdeb', 'GloVe-Fdeb-wh']
    elif emb_type == 'bert':
        paths.append(f'{org_dir}/{task}_BERT.json')
        paths.append(f'{org_dir}/{task}_BERT-wh.json')
        paths.append(f'{org_dir}/{task}_BERT-Fdeb.json')
        paths.append(f'{org_dir}/{task}_BERT-Fdeb-wh.json')
        labels = ['BERT', 'BERT-wh', 'BERT-Fdeb', 'BERT-Fdeb-wh']

    score_and_rank_dics = []
    for path in paths:
        score_and_rank_dic = json.load(open(path))
        score_and_rank_dics.append(score_and_rank_dic)

    out_name = f'{args.out_dir}/{task}_all_{args.emb_type}.png'
    plot(1000, score_and_rank_dics, labels, args, out_name=out_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_type', type=str)
    parser.add_argument('--json_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    main()
