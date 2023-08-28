from extract_bvp.Eluoren.chrom import chrom
from extract_bvp.Eluoren.ica import ica,project_ica
from extract_bvp.Eluoren.pos import pos,project_pos
from extract_bvp.Eluoren.pca import pca



def get_bvp_signal(data, fps, order, interval_length, chn=1, method='chrom'):
    assert method in ['chrom', 'pos', 'project_pos', 'ica', 'pjica', 'pca', 'single_chn'], 'Unknown method specified !'

    # first stage: extract bvp signal from row video channel with classical methods.
    if method == 'chrom':
        signal = chrom(data, fps=fps, interval_length=interval_length)

    elif method == 'pos':
        signal = pos(data, fps=fps, l=interval_length, order=order)

    elif method == 'project_pos':
        signal = project_pos(data, fps=fps, order=order, l=interval_length)

    elif method == 'ica':
        signal = ica(data, fps=fps, order=order)

    elif method == 'pjica':
        signal = project_ica(data, fps=fps, order=order)

    elif method == 'pca':
        signal = pca(data, fps=fps, order=order)

    elif method == 'single_chn':
        signal = data[:, chn]

    else:
        signal = None
        print('Unknown method specified !')

    return signal