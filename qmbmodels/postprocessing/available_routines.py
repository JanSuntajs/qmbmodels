from .entropy import entro_ave, footer_entro, entro_post_analysis
from .entropy import footer_entro_analyse, \
    footer_entro_no_sample_averaging
from .rstat import footer_r_ave, r_ave
from .sff import get_sff, footer_sff, get_tau_thouless, footer_t_thouless


# Routines available for calculations:
# get_entro_ave: average entropy, sampling averaging performed
#
# get_entro_ave_samples: no sampling averaging perfomed, results
# returned are given for each disorder realization
#
# entro_analyse: analysis w.r.t. to the number of samples
# included where the samples are ordered according to the
# difference of their variance w.r.t. to the theoretic
# prediction for the variance
#
# get_r: r calculations, sampling averaging performed.
_routines_dict = {
    'get_entro_ave': [entro_ave, 'Entropy', 17, footer_entro],
    'get_entro_ave_samples': [entro_ave, 'Entropy', None,
                              footer_entro_no_sample_averaging],
    'entro_analyse': [entro_post_analysis, 'Entropy',
                      None, footer_entro_analyse],
    'get_r': [r_ave, 'r_data', 15, footer_r_ave],
    'get_sff': [get_sff, 'SFF_spectrum', 7, footer_sff],
    'get_tau_thouless': [get_tau_thouless, 'SFF_spectrum', 11,
                         footer_t_thouless]

}
